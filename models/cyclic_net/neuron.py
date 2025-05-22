from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn import Parameter


# Abstract class / interface defining what a computation neuron requires.
class Neuron(ABC):

    @property
    def get_ID(self) -> int:
        """
        A unique identifier for this neuron.
        """
        return None

    @property
    def get_output_dim(self) -> int:
        """
        The output dimension of this neuron.
        """
        return None

    @property
    def is_input_neuron(self) -> bool:
        """
        Whether this neuron needs input data from the task at hand (classifying images, etc)
        or only needs to take the outputs of its in-neighbor neurons.
        """
        return None

    @property
    def get_neighbor_dim(self) -> dict[int, int]:
        """Returns a map from ID of in-neighbors to output size of in-neighbor."""
        return None

    @property
    def get_params(self) -> Parameter:
        return None

    @abstractmethod
    def compute(self,
                neighbor_outputs: dict[int, Tensor],
                input_data: Tensor = None
                ) -> Tensor:
        """
        The forward pass computation of this neuron.
        :param neighbor_outputs: The outputs of in-neighbor neurons, formatted in a map from
        the neighbor's ID to a tensor.
        :param input_data: The input data to the task, if this neuron is an input neuron.
        :return: The output of this neuron computed from in-neighbor outputs and potential input.
        """
        return None