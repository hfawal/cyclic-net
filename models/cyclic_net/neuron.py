from abc import ABC, abstractmethod
from typing import Tuple, Dict, List, Any

from torch import Tensor
from torch.nn import Parameter


# Abstract class / interface defining what a computation neuron requires.
class Neuron(ABC):

    @property
    def ID(self) -> int:
        """
        A unique identifier for this neuron.
        """
        return self._ID

    @ID.setter
    def ID(self, value: Any):
        self.check_ID("ID", value)
        self._ID = value

    @property
    def inneighbor_dims(self) -> Dict[int, Tuple[int, ...]]:
        """
        The shape of the activations this neuron receives from each inneighbor.
        """
        return self._inneighbor_dims

    @inneighbor_dims.setter
    def inneighbor_dims(self, value: Any):
        self.check_shape_dict("inneighbor_dims", value)
        self._inneighbor_dims = value

    @property
    def output_dims(self) -> Dict[int, Tuple[int, ...]]:
        """
        The output shape of this neuron's output per outneighbor.
        """
        return self._output_dims

    @output_dims.setter
    def output_dims(self, value: Any):
        self.check_shape_dict("output_dims", value)
        self._output_dims = value

    @property
    def input_data_dim(self) -> Tuple[int, ...]:
        """
        Returns the shape of the input that this Neuron expects, if it's an input neuron.
        """
        return self._input_data_dim

    @input_data_dim.setter
    def input_data_dim(self, value: Any):
        self.check_shape_tuple("input_data_dim", value)
        self._input_data_dim = value

    @property
    def is_input_neuron(self) -> bool:
        """
        Whether this neuron needs input data from the task at hand (classifying images, etc)
        or only needs to take the outputs of its inneighbor neurons.
        """
        return self._is_input_neuron

    @is_input_neuron.setter
    def is_input_neuron(self, value: Any):
        self.check_bool("is_input_neuron", value)
        self._is_input_neuron = value

    def __init__(self,
                 ID: int,
                 inneighbor_dims: Dict[int, Tuple[int, ...]],
                 output_dims: Dict[int, Tuple[int, ...]],
                 input_data_dim: Tuple[int, ...],
                 is_input_neuron: bool
                 ):
        """
        Initializes the neuron with the given key fields. Subclasses of Neuron remain
        responsible for setting the params field and implementing compute().

        :param ID: A unique identifier for this neuron.
        :param inneighbor_dims: The shape of the activations this neuron receives from each of
        its inneighbors.
        :param output_dims: The output shape of this neuron's output per outneighbor.
        :param input_data_dim: Returns the shape of the input that this Neuron expects, if it's
        an input neuron.
        :param is_input_neuron: Whether this neuron needs input data from the task at hand
        (classifying images, etc.) or only needs to take the outputs of its inneighbor neurons.
        """

        self.ID = ID
        self.inneighbor_dims = inneighbor_dims
        self.output_dims = output_dims
        self.input_data_dim = input_data_dim
        self.is_input_neuron = is_input_neuron

    @property
    @abstractmethod
    def params(self) -> List[Parameter]:
        """
        Returns a list of the learnable parameters of this neuron.
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
        will correspondingly have n as their first dimension.
        :return: The output of this neuron computed from in-neighbor outputs and potential input.
        Formatted as a dictionary of outneighbor IDs and the output vector for that outneighbor.
        """
        pass



    # ------------------------------------- Helpers ---------------------------------------------- #

    def check_ID(self, name: str, value: Any):
        if not isinstance(value, int):
            self.error(name, "of type int", type(value))
        if value < -2:
            self.error(name, ">= -1", value)

    def check_bool(self, name: str, value: Any):
        if not isinstance(value, bool):
            self.error(name, "of type bool", type(value))

    def check_shape_tuple(self, name: str, value: Any):
        if not isinstance(value, tuple):
            self.error(name, "of type tuple", type(value))
        if len(value) == 0:
            self.error(name, "non-empty", value)
        for elem in value:
            if not isinstance(elem, int):
                self.error(name + " tuple element", "of type int", type(elem))
            if elem <= 0:
                self.error(name + " tuple element", "positive", elem)

    def check_shape_dict(self, name: str, value: Any):
        if not isinstance(value, dict):
            self.error(name, "of type dict", type(value))
        if len(value) == 0:
            self.error(name, "non-empty", value)
        for ID, shape in value.items():
            self.check_ID(name + " key", ID)
            self.check_shape_tuple(name + " value", shape)

    def error(self, name: str, expected: str, value: Any):
        raise ValueError(f"Expected {name} to be {expected}, found {value}.")
