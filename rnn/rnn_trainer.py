import torch
import torch.nn as nn

class RNNTrainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, test_loader, device, max_grad_norm=1.0):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.max_grad_norm = max_grad_norm  # Maximum norm for gradient clipping

    def validate(self):
        self.model.eval()
        total_loss = 0
        num_correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_loader):
                data, target = data.to(self.device), target.to(self.device)
                num_outs = data.shape[1]
                outputs = torch.zeros(target.shape[0], num_outs, self.model.output_dim).to(self.device)
                h = self.model.init_hidden(target.shape[0])
                for i in range(num_outs):
                    output, h = self.model(data[:, i, :], h)
                    outputs[:, i, :] = output
                non_zeros = ~(data == 0.0).all(dim=2)
                non_zeros = non_zeros.cumsum(dim=1)
                last_indicices = non_zeros.argmax(dim=1)
                outputs = outputs[torch.arange(target.shape[0]), last_indicices, :]
                if self.model.output_dim == 1:
                    target = target.unsqueeze(1)
                loss = self.criterion(outputs, target)
                total_loss += loss.item()
                outputs = outputs > 0.5
                outputs = outputs.float()
                target = target.float()
                num_correct += (outputs == target).sum().item()
            return total_loss / len(self.val_loader), num_correct / (len(self.val_loader)*self.val_loader.batch_size)

                

    def train(self, num_epochs, save_interal = 1):
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            self.model.train()
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                num_outs = data.shape[1]
                outputs = torch.zeros(target.shape[0], num_outs, self.model.output_dim).to(self.device)
                h = self.model.init_hidden(data.shape[0])
                for i in range(num_outs):
                    output, h = self.model(data[:, i, :], h)
                    outputs[:, i, :] = output
                non_zeros = ~(data == 0.0).all(dim=2)
                non_zeros = non_zeros.cumsum(dim=1)
                last_indicices = non_zeros.argmax(dim=1)
                if self.model.output_dim == 1:
                    target = target.unsqueeze(1)
                loss = self.criterion(outputs[torch.arange(target.shape[0]), last_indicices, :], target)
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            val_loss, val_acc = self.validate()
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            if (epoch+1) % save_interal == 0:
                torch.save(self.model.state_dict(), f"models/rnn/model_{epoch+1}.pth")

    def test(self):
        self.model.eval()
        total_loss = 0
        num_correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.device), target.to(self.device)
                num_outs = data.shape[1]
                outputs = torch.zeros(target.shape[0], num_outs, self.model.output_dim).to(self.device)
                h = self.model.init_hidden(target.shape[0])
                for i in range(num_outs):
                    output, h = self.model(data[:, i, :], h)
                    outputs[:, i, :] = output
                non_zeros = ~(data == 0.0).all(dim=2)
                non_zeros = non_zeros.cumsum(dim=1)
                last_indicices = non_zeros.argmax(dim=1)
                outputs = outputs[torch.arange(target.shape[0]), last_indicices, :]
                if self.model.output_dim == 1:
                    target = target.unsqueeze(1)
                loss = self.criterion(outputs, target)
                total_loss += loss.item()
                outputs = outputs > 0.5
                outputs = outputs.float()
                target = target.float()
                num_correct += (outputs == target).sum().item()
            return total_loss / len(self.test_loader), num_correct / (len(self.test_loader)*self.test_loader.batch_size)




