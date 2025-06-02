from typing import Dict, Tuple

import torch
import torch.nn as nn

from models.cyclic_net.cyclic_net_sequence import CyclicNetSequence
from models.cyclic_net.cyclic_trainer_sequence import CyclicTrainerSequence
from models.cyclic_net.neuron import Neuron
from utils.contrastive_ibdm import ContrastiveIBDM
import yaml
import os

from graphs.complete_4n_imdb import build_linear_relu_neurons, build_readout_layer, get_lrs, get_thresholds

if __name__ == "__main__":
    print(f"Working directory: {os.getcwd()}")
    config = "cyclic_sequence.yaml"  # Path to your YAML config file
    with open(f"./experiments/configs/{config}", "r") as f:
        config = yaml.safe_load(f)

    # Build required neurons and readout node
    neurons: Dict[int, Neuron] = build_linear_relu_neurons()
    readout_neuron: Neuron = build_readout_layer()

    # Initialize the model and device
    model = CyclicNetSequence(neurons, config["num_iter"], readout_neuron)
    criterion = getattr(nn, config["criterion"])()
    optimizer = config["optimizer"]
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loader parameters from config
    data_loader = ContrastiveIBDM(batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=config["shuffle"])
    train_loader, val_loader, test_loader = data_loader.load_data(
        val_size=config["val_size"],
    )

    lrs: Tuple[Dict[int, float], float] = get_lrs()
    thresholds: Dict[int, float] = get_thresholds()

    # Initialize the trainer
    trainer = CyclicTrainerSequence(model, criterion, optimizer,
                            train_loader, val_loader, test_loader, device,
                            init_lr=lrs[0], readout_init_lr=lrs[1],
                            thresholds=thresholds)

    # Save the config file in the save directory
    os.makedirs(config["save_dir"], exist_ok=True)
    save_path = os.path.join(".", config["save_dir"])
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "config_final_model.yaml"), "w") as f_out:
        yaml.dump(config, f_out)

    # Train the model
    print("Begin training")
    # print(f"Validation Accuracy: {trainer.validate():.2f}%")

    trainer.train(num_epochs=config["num_epochs"], save_interval=config["save_interval"], save_dir=config["save_dir"])  # Adjust the number of epochs as needed
    print(f"Test Accuracy: {trainer.test():.2f}%")

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_path, "final_model.pt"))
