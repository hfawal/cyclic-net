import torch
import torch.nn as nn
import torch.optim as optim

from models.simple_nn.simple_nn import SimpleNN
from bp_trainer import BPTrainer
from utils.data_loader import MnistDataloader
import yaml
import os

if __name__ == "__main__":
    config = "simplenn.yaml"  # Path to your YAML config file
    with open(f"../configs/{config}", "r") as f:
        config = yaml.safe_load(f)

    # Initialize the model, criterion, optimizer, and device
    model = SimpleNN()
    criterion = getattr(nn, config["criterion"])()
    optimizer = getattr(optim, config["optimizer"])(model.parameters(), lr=config["lr"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loader parameters from config
    data_loader = MnistDataloader(
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        num_workers=config["num_workers"]
    )
    train_loader, val_loader, test_loader = data_loader.load_data(val_size=config["val_size"])

    # Initialize the trainer (assuming BPTrainer is defined in utils/trainers.py)
    trainer = BPTrainer(model, criterion, optimizer, train_loader, val_loader, test_loader, device)

    # Save the config file in the save directory
    os.makedirs(config["save_dir"], exist_ok=True)
    save_path = os.path.join("../models", config["save_dir"])
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "config_final_model.yaml"), "w") as f_out:
        yaml.dump(config, f_out)

    # Train the model
    trainer.train(num_epochs=config["num_epochs"], save_interval = 5, path = config["save_dir"])  # Adjust the number of epochs as needed
    trainer.test()

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_path, "final_model.pt"))