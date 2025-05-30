import torch
import torch.nn as nn
from rnn.model import RNN
from rnn.rnn_trainer import RNNTrainer
from utils.ibdm_data_loader import IMDBDataLoader





if __name__ == "__main__":
    dataloader = IMDBDataLoader(batch_size=256)
    train_data, val_data, test_data = dataloader.load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RNN(100, [100], [100], [100], 100, output_dim=1, device=device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    trainer = RNNTrainer(model, criterion, optimizer, train_data, val_data, test_data, device)
    trainer.train(10, save_interal=2)
