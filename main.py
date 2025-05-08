import torch
import torch.nn as nn
import torch.optim as optim
from utils.trainers import BPTrainer
from utils.data_loader import MnistDataloader
import yaml
import os

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.log_softmax(x, dim=1)  # Log softmax for multi-class classification
        return x

if __name__ == "__main__":
    config = "simplenn.yaml"  # Path to your YAML config file
    with open(f"configs/{config}", "r") as f:
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
    save_path = os.path.join("models", config["save_dir"])
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "config.yaml"), "w") as f_out:
        yaml.dump(config, f_out)

    # Train the model
    trainer.train(num_epochs=config["num_epochs"], save_interval = 5, path = config["save_dir"])  # Adjust the number of epochs as needed
    trainer.test()

    #save final model
    torch.save(model.state_dict(), os.path.join(save_path, "final_model.pt"))