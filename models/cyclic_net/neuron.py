from abc import ABC, abstractmethod
from typing import Tuple, Dict, List

from torch import Tensor
from torch.nn import Parameter


# Abstract class / interface defining what a computation neuron requires.
class Neuron(ABC):

    @property
    @abstractmethod
    def ID(self) -> int:
        """
        A unique identifier for this neuron.
        """
        pass

    @property
    @abstractmethod
    def output_dims(self) -> Dict[int, Tuple[int, ...]]:
        """
        The output shape of this neuron's output per outneighbor.
        """
        pass

    @property
    @abstractmethod
    def is_input_neuron(self) -> bool:
        """
        Whether this neuron needs input data from the task at hand (classifying images, etc)
        or only needs to take the outputs of its inneighbor neurons.
        """
        pass

    @property
    @abstractmethod
    def input_data_dim(self) -> Tuple[int, ...]:
        """
        Returns the shape of the input that this Neuron expects, if it's an input neuron.
        """
        pass

    @property
    @abstractmethod
    def inneighbor_dims(self) -> Dict[int, Tuple[int, ...]]:
        """
        Returns a map from ID of inneighbors to output shape of inneighbor.
        """
        pass

    @property
    @abstractmethod
    def params(self) -> List[Parameter]:
        """
        Returns the learnable parameters of this neuron.
        """
        pass

    @abstractmethod
    def compute(self,
                neighbor_outputs: dict[int, Tensor],
                input_data: Tensor = None
                ) -> Dict[int, Tensor]:
        """
        The forward pass computation of this neuron.
        :param neighbor_outputs: The outputs of in-neighbor neurons, formatted in a map from
        the neighbor's ID to a tensor.
        :param input_data: The input data to the task, if this neuron is an input neuron. The shape
        should be (n, <anything>) where n is the number of examples. Tensors in the output
        will correspondingly have n as their last dimension.
        :return: The output of this neuron computed from in-neighbor outputs and potential input.
        Formatted as a dictionary of outneighbor IDs and the output vector for that outneighbor.
        """
        pass