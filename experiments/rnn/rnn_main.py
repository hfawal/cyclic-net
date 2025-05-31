import torch
import torch.nn as nn
from models.rnn.model import RNN
from models.rnn.rnn_trainer import RNNTrainer
from utils.ibdm_data_loader import IMDBDataLoader



train = False

if __name__ == "__main__":
    #train the model
    if train:
        dataloader = IMDBDataLoader(batch_size=256)
        train_data, val_data, test_data = dataloader.load_data()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = RNN(100, [100], [100], [100], 100, output_dim=1, device=device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.to(device)
        trainer = RNNTrainer(model, criterion, optimizer, train_data, val_data, test_data, device)
        trainer.train(10, save_interal=2)

    else:
        #test the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RNN(100, [100], [100], [100], 100, output_dim=1, device=device)
        model.load_state_dict(torch.load("saves/model_8.pth"))
        model.to(device)
        dataloader = IMDBDataLoader(batch_size=256)
        train_data, val_data, test_data = dataloader.load_data()
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = RNNTrainer(model, criterion, optimizer, train_data, val_data, test_data, device)
        test_loss, test_acc = trainer.test()
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
