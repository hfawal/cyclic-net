import torch
import torch.nn as nn
from models.chain_ff.chain_ff import ChainFF
from models.chain_ff.ff_trainer import FFTrainer
from utils.data_loader import MnistDataloader
import yaml
import os

if __name__ == "__main__":
    config = "chainff.yaml"  # Path to your YAML config file
    with open(f"../configs/{config}", "r") as f:
        config = yaml.safe_load(f)

    # Initialize the model and device
    model = ChainFF()
    criterion = getattr(nn, config["criterion"])()
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loader parameters from config
    data_loader = MnistDataloader(
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        num_workers=config["num_workers"]
    )
    train_loader, val_loader, test_loader = data_loader.load_data(val_size=config["val_size"])

    # Initialize the trainer
    trainer = FFTrainer(model, criterion, train_loader, val_loader, test_loader, device, learning_rate=config["lr"], threshold=config["threshold"])

    # Save the config file in the save directory
    os.makedirs(config["save_dir"], exist_ok=True)
    save_path = os.path.join(".", config["save_dir"])
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "config_final_model.yaml"), "w") as f_out:
        yaml.dump(config, f_out)

    # Train the model
    trainer.train(num_epochs=config["num_epochs"])  # Adjust the number of epochs as needed
    trainer.test()

    # Save final model
    torch.save(model.state_dict(), os.path.join(save_path, "final_model.pt"))