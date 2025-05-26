from typing import Tuple, Dict, Set

import torch
import torch.nn as nn
from torch import Tensor

from models.cyclic_net.neuron import Neuron


class CyclicNet(nn.Module):

    def __init__(self,
                 neurons: Dict[int, Neuron],
                 number_iterations: int,
                 readout_neuron: Neuron,
                 ):
        """
        Defines a cyclic network where the neurons are given as a map of unique integer IDs
        (must be zero-indexed) to Neuron objects, and the graph is implicit in the output_dims
        and inneighbor_dims dictionaries present as properties of every Neuron object.
        Neurons propagate their outputs for a specified number of iterations before we read
        a particular value from the graph. The readout neuron is assigned the unique ID -1, and
        should not appear anywhere in the graph structure implicitly given, except as an outneighbor
        of certain neurons at the user's discretion. It is functionally a "sink" node
        that does not participate in propagation, with inneighbors specified through the object.
        The readout neuron itself should output to a nonexistent neuron with ID -2, which we treat
        as the overall output of the cylic net.

        :param neurons: A mapping of zero-indexed neuron IDs to Neuron objects.
        :param number_iterations: The number of iterations this cyclic net propagates outputs.
        :param readout_neuron: The readout neuron with ID -1 (not appearing in graph).
        """

        super(CyclicNet, self).__init__()
        self.neurons = neurons
        self.number_iterations = number_iterations
        self.readout_neuron = readout_neuron


    def forward(self,
                x: Tensor
                ) -> Tensor:
        """
        Computes the forward pass of this cyclic network given the input data.

        :param x: Input data tensor, with shape (<anything>, n) where n is the number of examples
        to evaluate the network on. Even when there is only 1 example, we require n to be present.

        :return: The prediction of this network on each example in x. This is a tensor with shape
        (<anything>, n).
        """

        activations_sink = None
        next_activations_source = None
        next_activations_sink = None

        # Propagate the neurons for some number of iterations.
        for i in range(self.number_iterations):

            next_activations_source, next_activations_sink = self.propagate(
                x, # Pass data.
                activations_sink, # Use previous activations.
                next_activations_source, # Source output dictionary.
                next_activations_sink # Sink output dictionary.
            )

            # Swap the dictionaries for the next iteration.
            activations_sink, next_activations_sink = next_activations_sink, activations_sink

        # Call the readout neuron with the final activations.
        result = self.readout_neuron.compute(activations_sink[-1])

        return result[-2]

    def propagate(self,
                  x: Tensor,
                  activations_sink: Dict[int, Dict[int, Tensor]] = None,
                  next_activations_source: Dict[int, Dict[int, Tensor]] = None,
                  next_activations_sink: Dict[int, Dict[int, Tensor]] = None
                  ) -> Tuple[Dict[int, Dict[int, Tensor]], Dict[int, Dict[int, Tensor]]]:
        """
        Computes the activations of all neurons in the graph, based on previous activations and
        new input data, one time. Stores outputs in the provided dictionaries.

        :param x: The input data to the network at this time step. Shape: (<anything>, n) where
        n is the number of examples to evaluate the network on. Even when there is only 1 example,
        we require that n = 1 be explicitly present.
        :param activations_sink: The previous activations of all neurons in the network, organized
        as a sink-based dictionary. activations_sink[A][B] is the activation that the inneighbor
        neuron B passes to the neuron A. Defaults to zero tensors if not provided.
        :param next_activations_sink: This is an output parameter dictionary which will be mutated
        and returned, to save space. After all the neuron computations complete, the entry
        next_activations_sink[A][B] is the activation that inneighbor neuron B will pass to neuron
        A on the next iteration of propagate(). A new dictionary is created if not provided.
        :param next_activations_source: This is an output parameter dictionary which will be mutated
        and returned, to save space. After all the neuron computations complete, the entry
        next_activations_source[B][A] is the activation that neuron B will pass to outneighbor
        neuron A on the next iteration of propagate(). A new dictionary is created if not provided.

        :return: next_activations_source, next_activations_sink.
        """

        n: int = x.shape[-1]

        # Source maps capture the activations that a neuron sends to its outneighbors.
        # Indexed by map[source][outneighbor].
        # Sink maps capture the activations that a neuron receives from its inneighbors.
        # Indexed by map[sink][inneighbor]. (i.e. source map of the transpose graph)

        # Default initialize the output parameters of this function.
        if next_activations_source is None:
            next_activations_source = dict()

        if next_activations_sink is None:
            next_activations_sink = {
                ID: dict()
                for ID in self.neurons.keys()
            }
            next_activations_sink[-1] = dict()

        # Default initialize the input parameter of this function.
        if activations_sink is None:
            activations_sink = {
                ID: dict()
                for ID in self.neurons.keys()
            }
            # Generate the zeros for the regular neuron inneighbor dictionaries.
            for current in activations_sink.keys():
                for inneighbor, dim in self.neurons[current].inneighbor_dims.items():
                    activations_sink[current][inneighbor] = torch.zeros(dim + (n,))
            # Generate the zeros for the inneighbors of the readout neuron.
            activations_sink[-1] = dict()
            for inneighbor, dim in self.readout_neuron.inneighbor_dims.items():
                activations_sink[-1][inneighbor] = torch.zeros(dim + (n,))

        # Now compute neurons sequentially and write to the next activation dictionaries.
        # In theory this step should be parallelizable but due to the python GIL we would need to
        # use multiprocessing for any real gain, and this is unnecessary complexity for now.
        for current, neuron in self.neurons.items():

            # Detach the inneighbor activations from PyTorch's computation graph.
            for inneighbor, actvtn in activations_sink[current].items():
                activations_sink[current][inneighbor] = actvtn.detach()

            # Compute and set the outneighbor activations.
            next_activations_source[current] = neuron.compute(activations_sink[current], x)
            for outneighbor, actvtn in next_activations_source[current].items():
                next_activations_sink[outneighbor][current] = actvtn

        return next_activations_source, next_activations_sink




