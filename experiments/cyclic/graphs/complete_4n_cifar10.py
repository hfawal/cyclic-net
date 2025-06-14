from typing import Dict, Tuple

from models.cyclic_net.neuron import Neuron
from models.cyclic_net.neurons.HeterogeneousLinearReLU import HeterogeneousLinearReLU
from models.cyclic_net.neurons.HomogeneousLinearReLU import HomogeneousLinearReLU
from models.cyclic_net.neurons.LinearLogSoftmax import LinearLogSoftmax

# Each node sends a (10,) vector to every other node (except itself)
adjacency_list: Dict[int, Dict[int, Tuple[int, ...]]] = {
    0: {1: (2000,), 2: (2000,), 3: (2000,), -1: (2000,)},
    1: {0: (2000,), 2: (2000,), 3: (2000,), -1: (2000,)},
    2: {0: (2000,), 1: (2000,), 3: (2000,), -1: (2000,)},
    3: {0: (2000,), 1: (2000,), 2: (2000,), -1: (2000,)}
}

# Readout node takes in outputs from all nodes
readout_inneighbor_dims: Dict[int, Tuple[int, ...]] = {
    0: (2000,),
    1: (2000,),
    2: (2000,),
    3: (2000,)
}

init_lr: Dict[int, float] = {
    0: 0.1,
    1: 0.1,
    2: 0.1,
    3: 0.1,
}

thresholds: Dict[int, float] = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
}

readout_init_lr: float = 0.01


def invert_graph() -> Dict[int, Dict[int, Tuple[int, ...]]]:
    inneighbors = {}

    for src, dests in adjacency_list.items():
        for dst, shape in dests.items():
            if dst not in inneighbors:
                inneighbors[dst] = {}
            inneighbors[dst][src] = shape

    return inneighbors


def build_linear_relu_neurons() -> Dict[int, Neuron]:
    inneighbor_dims = invert_graph()
    neurons = {}
    for nid in adjacency_list:
        neurons[nid] = HomogeneousLinearReLU(
            ID=nid,
            inneighbor_dims=inneighbor_dims[nid],
            output_dims=adjacency_list[nid],
            input_data_dim=(3082,),
            is_input_neuron=True,
        )
    return neurons


def build_readout_layer() -> Neuron:
    return LinearLogSoftmax(
        ID=-1,
        inneighbor_dims=readout_inneighbor_dims,
        output_dims={-2: (10,)},
        input_data_dim=(3082,),
        is_input_neuron=False,
    )


def get_lrs() -> Tuple[Dict[int, float], float]:
    return init_lr, readout_init_lr


def get_thresholds() -> Dict[int, float]:
    return thresholds
