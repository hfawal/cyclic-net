from typing import Dict, Tuple, List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.cyclic_net.neuron import Neuron


class ConvLogSoftmax(Neuron):
    """
    A convolutional neuron that produces a log-softmax output for classification.

    This neuron aggregates input feature maps from its inneighbors, optionally processes raw input data
    and associated labels (if it is an input neuron), and outputs class logits via a convolution followed
    by adaptive average pooling and log-softmax activation. This architecture is suitable for use as a
    readout layer for classification tasks such as CIFAR-10.

    Attributes:
        _input_data_dim (Tuple[int, ...]): Shape of the input image data (C, H, W).
        _label_dim (Tuple[int, ...]): Shape of the label input (e.g., one-hot encoding).
        label_project (nn.Linear): Linear layer to project label input to a spatial map.
        input_conv (nn.Conv2d): Convolutional layer to process the input image.
        input_label_conv (nn.Conv2d): Convolutional layer to fuse image and label features.
        conv (nn.Conv2d): Main convolutional layer to generate class logits.
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
                 neuron_out_channels: int = 10,
                 neuron_kernel_size: int = 1,
                 neuron_padding: int = 0) -> None:
        # Store input data and label shapes
        self._input_data_dim = input_data_dim
        self._label_dim = label_dim
        super().__init__(ID, inneighbor_dims, output_dims, input_data_dim, is_input_neuron)

        # Compute total number of input channels from inneighbors
        in_channels = sum(dim[0] for dim in inneighbor_dims.values())

        if is_input_neuron:
            # Add label fusion output to channel count
            in_channels += input_label_out_channels

            # Project label vector to spatial dimensions
            self.label_project = nn.Linear(label_dim[0], label_out_channels * input_data_dim[1] * input_data_dim[2])

            # Process raw image input
            self.input_conv = nn.Conv2d(
                in_channels=input_data_dim[0],
                out_channels=input_out_channels,
                kernel_size=input_kernel_size,
                padding=input_padding)

            # Fuse image and label projections
            self.input_label_conv = nn.Conv2d(
                in_channels=input_out_channels + label_out_channels,
                out_channels=input_label_out_channels,
                kernel_size=input_label_kernel_size,
                padding=input_label_padding)

        # Final convolution to produce class logits
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=neuron_out_channels,
            kernel_size=neuron_kernel_size,
            padding=neuron_padding)

    @property
    def params(self) -> List[Parameter]:
        # Return all parameters registered in this module
        return list(self.parameters())

    def compute(self,
                neighbor_outputs: Dict[int, Tensor],
                input_data: Tensor = None) -> Dict[int, Tensor]:
        # Determine the device to perform computations on
        device: torch.device = self.conv.weight.device

        # Normalize and gather outputs from all inneighbors
        inputs: List[Tensor] = [
            F.normalize(neighbor_outputs[nid].to(device), p=2, dim=1)
            for nid in sorted(neighbor_outputs)
        ]

        # If input neuron, process raw input and label
        if self.is_input_neuron and input_data is not None:
            # Split and reshape input data and label
            input_data_dim_flat: int = int(torch.prod(torch.tensor(self._input_data_dim)))
            raw_input: Tensor = input_data[:, :input_data_dim_flat].view(-1, *self._input_data_dim).to(device)
            label_input: Tensor = input_data[:, input_data_dim_flat:].view(-1, *self._label_dim).to(device)

            # Apply initial conv to raw input
            input_out: Tensor = self.input_conv(raw_input)

            # Project label and reshape to spatial map
            label_proj: Tensor = self.label_project(label_input).view(
                -1, 1, self._input_data_dim[1], self._input_data_dim[2]
            )

            # Concatenate and fuse input + label
            combined: Tensor = torch.cat([input_out, label_proj], dim=1)
            inputs.append(F.relu(self.input_label_conv(combined)))

        # Concatenate all inputs into one tensor along the channel dimension
        x: Tensor = torch.cat(inputs, dim=1)

        # Apply final convolution
        out: Tensor = self.conv(x)

        # Pool spatial dimensions to 1x1 to produce global representation
        out = F.adaptive_avg_pool2d(out, (1, 1))

        # Flatten pooled output to shape [B, num_classes]
        out = out.view(out.size(0), -1)

        # Apply log-softmax for classification
        out = F.log_softmax(out, dim=1)

        # Return the same output to all downstream neurons
        return {nid: out for nid in self.output_dims}
