from typing import Dict, Tuple, List, Any

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, ParameterDict
from models.cyclic_net.neuron import Neuron


class HomogeneousLinearReLU(Neuron):
    """
    A neuron where all outneighbors receive the same output.
    It applies a linear transformation followed by ReLU activation to the normalized,
    concatenated inputs from its inneighbors (and optionally input data if it is an input neuron).

    Only one shared weight and bias are used for the linear layer, reducing parameter count and ensuring
    uniform output across all connected neighbors.

    This class assumes that all outneighbors expect the same output shape.
    """
    def __init__(self,
                 ID: int,
                 inneighbor_dims: Dict[int, Tuple[int, ...]],
                 output_dims: Dict[int, Tuple[int, ...]],
                 input_data_dim: Tuple[int, ...],
                 is_input_neuron: bool,
                 ):
        super().__init__(ID, inneighbor_dims, output_dims, input_data_dim, is_input_neuron)

        # Sort inneighbor IDs to ensure consistent input ordering
        sorted_nids = sorted(inneighbor_dims)

        # Compute total input dimension by flattening each input and summing
        total_in_dim = sum(int(torch.prod(torch.tensor(inneighbor_dims[nid]))) for nid in sorted_nids)
        if is_input_neuron:
            total_in_dim += int(torch.prod(torch.tensor(input_data_dim)))

        # Use one shared output shape (assume all are the same)
        out_shape = next(iter(output_dims.values()))
        output_dim = int(torch.prod(torch.tensor(out_shape)))

        # Single shared weight & bias
        self.weight = Parameter(torch.randn(output_dim, total_in_dim) * 0.01)
        self.bias = Parameter(torch.zeros(output_dim))

    @property
    def params(self) -> List[Parameter]:
        return [self.weight, self.bias]

    @Neuron.output_dims.setter
    def output_dims(self, value: Any):
        # Call the superclass validation
        super(self.__class__, self.__class__).output_dims.fset(self, value)

        # Additional check: ensure all output shapes are the same
        shapes = list(value.values())
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(f"All output_dims must have the same shape, but got: {shapes}")

    def compute(self, neighbor_outputs: dict[int, Tensor], input_data: Tensor = None) -> dict[int, Tensor]:
        device = self.weight.device
        inputs = [
            F.normalize(neighbor_outputs[nid].to(device), p=2, dim=1)
            for nid in sorted(neighbor_outputs)
        ]
        if self.is_input_neuron and input_data is not None:
            inputs.append(F.normalize(input_data.to(device), p=2, dim=1))

        x = torch.cat(inputs, dim=1)
        activated = F.relu(F.linear(x, self.weight, self.bias))

        # Return same output for each outneighbor
        return {nid: activated for nid in self.output_dims}
