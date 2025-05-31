from typing import Dict, Tuple, List
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, ParameterDict
from models.cyclic_net.neuron import Neuron


class LinearLogSoftmax(Neuron):
    """
    A neuron that applies a linear transformation followed by log_softmax.
    The inputs are the concatenated outputs of its inneighbors (flattened and ordered by ID).
    The output is a single vector which is partitioned into different slices for each outneighbor.
    """
    def __init__(self,
                 ID: int,
                 inneighbor_dims: Dict[int, Tuple[int, ...]],
                 output_dims: Dict[int, Tuple[int, ...]],
                 input_data_dim: Tuple[int, ...],
                 is_input_neuron: bool,
                 ):
        super().__init__(ID, inneighbor_dims, output_dims, input_data_dim, is_input_neuron)

        sorted_nids = sorted(inneighbor_dims)
        total_in_dim = sum(int(torch.prod(torch.tensor(inneighbor_dims[nid]))) for nid in sorted_nids)
        if is_input_neuron:
            total_in_dim += int(torch.prod(torch.tensor(input_data_dim)))

        self._weights = ParameterDict()
        self._biases = ParameterDict()
        for nid in sorted(output_dims):
            out_dim = int(torch.prod(torch.tensor(output_dims[nid])))
            self._weights[str(nid)] = Parameter(torch.randn(out_dim, total_in_dim) * 0.01)
            self._biases[str(nid)] = Parameter(torch.zeros(out_dim))

    @property
    def params(self) -> List[Parameter]:
        return list(self._weights.values()) + list(self._biases.values())

    def compute(self,
                neighbor_outputs: dict[int, Tensor],
                input_data: Tensor = None
                ) -> dict[int, Tensor]:
        device = next(iter(self._weights.values())).device
        inputs = [
            F.normalize(neighbor_outputs[nid].to(device), p=2, dim=1)
            for nid in sorted(neighbor_outputs)
        ]
        if self.is_input_neuron and input_data is not None:
            inputs.append(input_data.to(device))
        x = torch.cat(inputs, dim=1)

        outputs = {}
        for nid in self.output_dims:
            weight = self._weights[str(nid)]
            bias = self._biases[str(nid)]
            linear_output = F.linear(x, weight, bias)
            outputs[nid] = F.log_softmax(linear_output, dim=1)
        return outputs
