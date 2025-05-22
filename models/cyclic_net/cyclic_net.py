from typing import Tuple, Dict, Set

import torch
import torch.nn as nn
from torch import Tensor

from models.cyclic_net.neuron import Neuron


class CyclicNet(nn.Module):

    def __init__(self,
                 neurons: Dict[int, Neuron],
                 graph: Dict[int, Set[int]],
                 number_iterations: int,
                 readout_neuron: Neuron,
                 readout_inneighbors: Set[int]
                 ):
        """
        Defines a cyclic network where the neurons are given as a map of unique integer IDs
        (must be zero-indexed) to Neuron objects, and the graph is the given adjacency list.
        Neurons propagate their outputs for a specified number of iterations before we read
        a particular value from the graph. The readout neuron is assigned the unique ID -1, and
        should not appear anywhere in the graph structure given. It is functionally a "sink" node
        that does not participate in propagation, with inneighbors specified through the parameter.
        :param neurons: A mapping of zero-indexed neuron IDs to Neuron objects.
        :param graph: An adjacency list representation of a graph defined using neuron IDs.
        :param number_iterations: The number of iterations this cyclic net propagates outputs.
        :param readout_neuron: The readout neuron with ID -1 (not appearing in graph).
        :param readout_inneighbors: The IDs of the inneighbor neurons of the readout neuron.
        """
        super(CyclicNet, self).__init__()
        self.neurons = neurons
        self.graph = graph
        self.number_iterations = number_iterations
        self.readout_neuron = readout_neuron
        self.readout_inneighbors = readout_inneighbors


    def forward(self, x: Tensor):
        """
        TODO: write this docstring
        :param x:
        :return:
        """

    def propagate(self,
                  x: Tensor,
                  activations_sink: Dict[int, Dict[int, Tensor]] = None,
                  next_activations_sink: Dict[int, Dict[int, Tensor]] = None,
                  next_activations_source: Dict[int, Dict[int, Tensor]] = None
                  ) -> Tuple[Dict[int, Dict[int, Tensor]], Dict[int, Dict[int, Tensor]]]:
        """
        TODO: write this docstring
        :param x:
        :param activations_sink:
        :param next_activations_sink: OUTPUT PARAM, MUTATED
        :param next_activations_source: OUTPUT PARAM, MUTATED
        :return:
        """
        # The shape of x should be (<anything>, n). Here, n is the number of examples.
        # Even when there is only 1 example, n = 1 should be explicitly present.
        n: int = x.shape[-1]

        # Source maps capture the activations that a neuron sends to its outneighbors.
        # Indexed by map[source][neighbor].
        # Sink maps capture the activations that a neuron receives from its inneighbors.
        # Indexed by map[sink][inneighbor]. (i.e. source map of the transpose graph)

        # Default initialize output params.
        if next_activations_source is None:
            next_activations_source = dict()
        if next_activations_sink is None:
            next_activations_sink = {
                ID: dict()
                for ID in self.neurons.keys()
            }
            next_activations_sink[-1] = dict()


        if activations_sink is None:
            # Generate the zeros for the input dictionaries.
            activations_sink = {
                ID: dict()
                for ID in self.neurons.keys()
            }
            for current in activations_sink.keys():
                for inneighbor, dim in self.neurons[current].inneighbor_dims.items():
                    activations_sink[current][inneighbor] = torch.zeros(dim + (n,))
            # Inneighbors of the readout neuron.
            activations_sink[-1] = dict()
            for inneighbor, dim in self.readout_neuron.inneighbor_dims.items():
                activations_sink[-1][inneighbor] = torch.zeros(dim + (n,))

        # Now compute neurons sequentially and write to the next activation dictionaries.
        # In theory this step should be parallelizable but due to the python GIL we would need to
        # use multiprocessing for any real gain, and this is unnecessary complexity for now.
        for current, neuron in self.neurons.items():
            next_activations_source[current] = neuron.compute(activations_sink[current], x)
            for outneighbor, output in next_activations_source[current].items():
                next_activations_sink[outneighbor][current] = output

        return next_activations_sink, next_activations_source




