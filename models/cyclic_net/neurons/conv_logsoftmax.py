from typing import Dict, Tuple, List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.cyclic_net.neuron import Neuron


class ConvLogSoftmax(Neuron):
    """
    A neuron that applies a convolution followed by log_softmax.
    The inputs are the concatenated outputs of its inneighbors (preserving spatial dimensions).
    The output is a multichannel feature map.
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
                 neuron_kernel_size: int = 1,
                 neuron_padding: int = 0) -> None:
        self._input_data_dim = input_data_dim
        self._label_dim = label_dim
        super().__init__(ID, inneighbor_dims, output_dims, input_data_dim, is_input_neuron)

        in_channels = sum(dim[0] for dim in inneighbor_dims.values())
        if is_input_neuron:
            in_channels += input_label_out_channels
            self.label_project = nn.Linear(label_dim[0], label_out_channels * input_data_dim[1] * input_data_dim[2])
            self.input_conv = nn.Conv2d(
                in_channels=input_data_dim[0],
                out_channels=input_out_channels,
                kernel_size=input_kernel_size,
                padding=input_padding)
            self.input_label_conv = nn.Conv2d(
                in_channels=input_out_channels + label_out_channels,
                out_channels=input_label_out_channels,
                kernel_size=input_label_kernel_size,
                padding=input_label_padding)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=neuron_out_channels,
            kernel_size=neuron_kernel_size,
            padding=neuron_padding)

    @property
    def params(self) -> List[Parameter]:
        return list(self.conv.parameters())

    def compute(self,
                neighbor_outputs: Dict[int, Tensor],
                input_data: Tensor = None) -> Dict[int, Tensor]:
        device: torch.device = self.conv.weight.device
        inputs: List[Tensor] = [
            F.normalize(neighbor_outputs[nid].to(device), p=2, dim=1)
            for nid in sorted(neighbor_outputs)
        ]

        if self.is_input_neuron and input_data is not None:
            input_data_dim_flat: int = int(torch.prod(torch.tensor(self._input_data_dim)))
            raw_input: Tensor = input_data[:, :input_data_dim_flat].view(-1, *self._input_data_dim).to(device)
            label_input: Tensor = input_data[:, input_data_dim_flat:].view(-1, *self._label_dim).to(device)

            input_out: Tensor = self.input_conv(raw_input)
            label_proj: Tensor = self.label_project(label_input).view(
                -1, 1, self._input_data_dim[1], self._input_data_dim[2]
            )
            combined: Tensor = torch.cat([input_out, label_proj], dim=1)
            inputs.append(F.relu(self.input_label_conv(combined)))

        x: Tensor = torch.cat(inputs, dim=1)
        out: Tensor = self.conv(x)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)  # Flatten
        out = F.log_softmax(out, dim=1)
        return {nid: out for nid in self.output_dims}
