from typing import Dict, Tuple, List, Any

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, ParameterDict
from models.cyclic_net.neuron import Neuron
import torch.nn as nn


class ConvReLU(Neuron):
    """
    A convolutional neuron that applies ReLU activation to inputs gathered from inneighbors.

    For input neurons:
        - The input data and label are split and reshaped.
        - The image is passed through a convolution layer.
        - The label is projected to a spatial map and combined with the image output.
        - The combination is convolved again and ReLU-activated.

    For all neurons:
        - The outputs of all inneighbors are normalized and concatenated along the channel axis.
        - A final convolution followed by ReLU is applied.
        - The result is distributed to all output neighbors.
    """
    def __init__(self,
                 ID: int,
                 inneighbor_dims: Dict[int, Tuple[int, ...]],
                 output_dims: Dict[int, Tuple[int, ...]],
                 input_data_dim: Tuple[int, ...],
                 is_input_neuron: bool,
                 label_dim: Tuple[int, ...],
                 input_out_channels: int = 1,
                 input_kernel_size: int = 3,
                 input_padding: int = 1,
                 label_out_channels: int = 1,
                 input_label_out_channels: int = 2,
                 input_label_kernel_size: int = 3,
                 input_label_padding: int = 1,
                 neuron_out_channels: int = 1,
                 neuron_kernel_size: int = 3,
                 neuron_padding: int = 1) -> None:
        self._input_data_dim = input_data_dim
        self._label_dim = label_dim
        super().__init__(ID, inneighbor_dims, output_dims, input_data_dim, is_input_neuron)

        # Sort inneighbor IDs to ensure consistent input ordering
        sorted_nids = sorted(inneighbor_dims)

        # Compute total input dimension by flattening each input and summing
        total_in_dim = sum(int(torch.prod(torch.tensor(inneighbor_dims[nid]))) for nid in sorted_nids)
        if is_input_neuron:
            total_in_dim += int(torch.prod(torch.tensor(input_data_dim))) + int(torch.prod(torch.tensor(label_dim)))

        self.input_data_dim = input_data_dim
        self.label_dim = label_dim

        # Compute in_channels: sum of inneighbor_dims channels, plus input_data_dim[0] if is_input_neuron
        in_channels = sum(dim[0] for dim in inneighbor_dims.values())
        if is_input_neuron:
            in_channels += input_label_out_channels
            # Add a linear projection layer for the label
            self.label_project = nn.Linear(label_dim[0], label_out_channels * input_data_dim[1] * input_data_dim[2])

            # Input image conv
            self.input_conv = nn.Conv2d(
                in_channels=input_data_dim[0],
                out_channels=input_out_channels,
                kernel_size=input_kernel_size,
                padding=input_padding)

            # Input image + label conv
            self.input_label_conv = nn.Conv2d(
                in_channels=input_out_channels + label_out_channels,
                out_channels=input_label_out_channels,
                kernel_size=input_label_kernel_size,
                padding=input_label_padding)

        # Final neighbor input Conv
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=neuron_out_channels,
            kernel_size=neuron_kernel_size,
            padding=neuron_padding)

    @property
    def params(self) -> List[Parameter]:
        if self.is_input_neuron:
            return list(self.conv.parameters()) + \
                list(self.label_project.parameters()) + \
                list(self.input_conv.parameters()) + \
                list(self.input_label_conv.parameters())
        return list(self.conv.parameters())

    @Neuron.output_dims.setter
    def output_dims(self, value: Dict[int, Tuple[int, ...]]) -> None:
        # Call the superclass validation
        super(self.__class__, self.__class__).output_dims.fset(self, value)

        # Additional check: ensure all output shapes are the same
        shapes = list(value.values())
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(f"All output_dims must have the same shape, but got: {shapes}")

    @Neuron.inneighbor_dims.setter
    def inneighbor_dims(self, value: Dict[int, Tuple[int, ...]]) -> None:
        # Call the superclass setter
        super(ConvReLU, self.__class__).inneighbor_dims.fset(self, value)

        # Check if all input shapes match self.input_data_dim except for the channel dimension
        shapes = list(value.values())
        if not all(len(shape) == len(self.input_data_dim) and shape[1:] == self.input_data_dim[1:] for shape in shapes):
            raise ValueError(f"All inneighbor_dims must match input_data_dim "
                             f"{self.input_data_dim} in spatial dimensions, but got: {shapes}")

    def compute(self, neighbor_outputs: Dict[int, Tensor], input_data: Tensor = None) -> Dict[int, Tensor]:
        # Get the device for computation
        device: torch.device = next(self.conv.parameters()).device

        # Normalize and collect outputs from all inneighbors
        inputs: List[Tensor] = [
            F.normalize(neighbor_outputs[nid].to(device), p=2, dim=1)
            for nid in sorted(neighbor_outputs)
        ]

        # If this is an input neuron, handle additional processing
        if self.is_input_neuron and input_data is not None:
            # Split and reshape raw input and label input
            input_data_dim_flat: int = int(torch.prod(torch.tensor(self.input_data_dim)))
            raw_input: Tensor = input_data[:, :input_data_dim_flat].view(-1, *self.input_data_dim).to(device)
            label_input: Tensor = input_data[:, input_data_dim_flat:].view(-1, *self.label_dim).to(device)

            # Apply convolution to raw image input
            input_out: Tensor = self.input_conv(raw_input)

            # Project label to spatial dimensions
            label_proj: Tensor = self.label_project(label_input).view(
                -1, 1, self.input_data_dim[1], self.input_data_dim[2]
            )

            # Concatenate and convolve image and label information
            combined: Tensor = torch.cat([input_out, label_proj], dim=1)
            inputs.append(F.relu(self.input_label_conv(combined)))

        # Concatenate all inputs along channel dimension
        neuron_input: Tensor = torch.cat(inputs, dim=1)

        # Apply the main convolution and ReLU activation
        activated: Tensor = F.relu(self.conv(neuron_input))

        # Return the result to all output neighbors
        return {nid: activated for nid in self.output_dims}
