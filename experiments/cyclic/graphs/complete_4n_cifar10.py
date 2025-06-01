from typing import Dict, Tuple

from models.cyclic_net.neuron import Neuron
from models.cyclic_net.neurons.HeterogeneousLinearReLU import HeterogeneousLinearReLU
from models.cyclic_net.neurons.HomogeneousLinearReLU import HomogeneousLinearReLU
from models.cyclic_net.neurons.LinearLogSoftmax import LinearLogSoftmax
from models.cyclic_net.neurons.conv_logsoftmax import ConvLogSoftmax
from models.cyclic_net.neurons.conv_relu import ConvReLU

# Each node sends a (10,) vector to every other node (except itself)
adjacency_list: Dict[int, Dict[int, Tuple[int, ...]]] = {
    0: {1: (1, 32, 32), 2: (1, 32, 32), 3: (1, 32, 32), -1: (1, 32, 32)},
    1: {0: (1, 32, 32), 2: (1, 32, 32), 3: (1, 32, 32), -1: (1, 32, 32)},
    2: {0: (1, 32, 32), 1: (1, 32, 32), 3: (1, 32, 32), -1: (1, 32, 32)},
    3: {0: (1, 32, 32), 1: (1, 32, 32), 2: (1, 32, 32), -1: (1, 32, 32)}
}

preprocess_neuron_outneighbor_dims: Dict[int, Tuple[int, ...]] = {
    0: (1, 32, 32),
    1: (1, 32, 32),
    2: (1, 32, 32),
    3: (1, 32, 32)
}


preprocess_neuron_id = 10

# Readout node takes in outputs from all nodes
readout_inneighbor_dims: Dict[int, Tuple[int, ...]] = {
    0: (1, 32, 32),
    1: (1, 32, 32),
    2: (1, 32, 32),
    3: (1, 32, 32)
}

init_lr: Dict[int, float] = {
    0: 0.001,
    1: 0.001,
    2: 0.001,
    3: 0.001,
    preprocess_neuron_id: 0.001,
}

thresholds: Dict[int, float] = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    preprocess_neuron_id: 1,
}

readout_init_lr: float = 0.001


def invert_graph() -> Dict[int, Dict[int, Tuple[int, ...]]]:
    inneighbors = {}

    for src, dests in adjacency_list.items():
        for dst, shape in dests.items():
            if dst not in inneighbors:
                inneighbors[dst] = {}
            inneighbors[dst][src] = shape

    for dst, shape in preprocess_neuron_outneighbor_dims.items():
        inneighbors[dst][preprocess_neuron_id] = shape

    return inneighbors


def build_linear_relu_neurons() -> Dict[int, Neuron]:
    inneighbor_dims = invert_graph()
    neurons = {}
    for nid in adjacency_list:
        neurons[nid] = ConvReLU(
            ID=nid,
            inneighbor_dims=inneighbor_dims[nid],
            output_dims=adjacency_list[nid],
            input_data_dim=(3, 32, 32),
            is_input_neuron=False,
            label_dim=(10,),
            input_out_channels=1,
            input_kernel_size=9,
            input_padding=4,
            label_out_channels=1,
            input_label_out_channels=2,
            input_label_kernel_size=9,
            input_label_padding=4,
            neuron_out_channels=10,
            neuron_kernel_size=9,
            neuron_padding=4
        )

    neurons[preprocess_neuron_id] = ConvReLU(
            ID=preprocess_neuron_id,
            inneighbor_dims={},
            output_dims=preprocess_neuron_outneighbor_dims,
            input_data_dim=(3, 32, 32),
            is_input_neuron=True,
            label_dim=(10,),
            input_out_channels=1,
            input_kernel_size=9,
            input_padding=4,
            label_out_channels=1,
            input_label_out_channels=2,
            input_label_kernel_size=9,
            input_label_padding=4,
            neuron_out_channels=10,
            neuron_kernel_size=9,
            neuron_padding=4
        )

    return neurons


def build_readout_layer() -> Neuron:
    return ConvLogSoftmax(
        ID=-1,
        inneighbor_dims=readout_inneighbor_dims,
        output_dims={-2: (10,)},
        input_data_dim=(3, 32, 32),
        is_input_neuron=False,
        label_dim=(10,),
        input_out_channels=1,
        input_kernel_size=9,
        input_padding=4,
        label_out_channels=1,
        input_label_out_channels=2,
        input_label_kernel_size=9,
        input_label_padding=4,
        neuron_out_channels=10,
        neuron_kernel_size=9,
        neuron_padding=4
    )


def get_lrs() -> Tuple[Dict[int, float], float]:
    return init_lr, readout_init_lr


def get_thresholds() -> Dict[int, float]:
    return thresholds
