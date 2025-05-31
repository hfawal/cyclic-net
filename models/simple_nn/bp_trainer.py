import torch
from matplotlib import pyplot as plt
class BPTrainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader, test_loader, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model.to(self.device)

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        return accuracy

    def train(self, num_epochs, save_interval, path):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for i, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            validation_accuracy = self.validate()
            if (epoch + 1) % save_interval == 0:
                torch.save(self.model.state_dict(), f'./models/{path}/model_epoch_{epoch+1}.pt')
                print(f'Model saved at epoch {epoch+1}')

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(self.train_loader):.4f}, '
                  f'Accuracy: {100 * correct / total:.2f}%, Validation Accuracy: {validation_accuracy:.2f}%')

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        images_shown = 0
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                for idx in range(min(5, images.size(0))):
                    img = images[idx].cpu().squeeze()
                    if img.ndim == 2:
                        axes[images_shown].imshow(img, cmap='gray')
                    else:
                        axes[images_shown].imshow(img.permute(1, 2, 0))
                    axes[images_shown].set_title(f'Pred: {predicted[idx].item()}, True: {labels[idx].item()}')
                    axes[images_shown].axis('off')
                    images_shown += 1
                    if images_shown == 5:
                        break
                if images_shown == 5:
                    break
        accuracy = 100 * correct / total
        print(f'Test Accuracy: {accuracy:.2f}%')
        plt.tight_layout()
        plt.show()