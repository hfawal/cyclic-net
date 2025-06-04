import math
import torch
from matplotlib import pyplot as plt


# Feedforward local learning trainer
class FFTrainer:
    def __init__(self, model, criterion, train_loader, val_loader, test_loader, device, learning_rate=1e-3, threshold=2.0):
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.lr = learning_rate
        self.threshold = threshold
        self.model.to(self.device)

    def compute_goodness(self, activations):
        return (activations ** 2).sum(dim=1)

    def train(self, num_epochs):
        results = {
            "validation_accuracy": [],
            "neuron_loss": {ID: [] for ID in self.model.neurons.keys()},
            "readout_loss": []
        }

        # torch.autograd.set_detect_anomaly(True)
        # current_lr = self.lr
        for epoch in range(num_epochs):
            current_lr = self.lr * 0.5 * (1 + math.cos(math.pi * epoch / num_epochs))
            print(f"Epoch {epoch + 1}/{num_epochs}, Learning Rate: {current_lr:.6f}")
            self.model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                images = images.view(images.size(0), -1)

                pos_inputs = torch.cat([images, torch.nn.functional.one_hot(labels, 10).float()], dim=1)

                neg_labels = (labels + torch.randint(1, 10, labels.shape, device=self.device)) % 10
                neg_inputs = torch.cat([images, torch.nn.functional.one_hot(neg_labels, 10).float()], dim=1)

                neu_inputs = torch.cat(
                    [images.view(images.size(0), -1), 0.1 * torch.ones((labels.size(0), 10), device=self.device)],
                    dim=1)

                activations_pos = self.model.forward_activations(pos_inputs)
                activations_neg = self.model.forward_activations(neg_inputs)

                for i, layer in enumerate([self.model.fc1, self.model.fc2, self.model.fc3, self.model.fc4]):

                    g_pos = self.compute_goodness(activations_pos[i])
                    g_neg = self.compute_goodness(activations_neg[i])

                    loss = -torch.log(torch.sigmoid(g_pos - self.threshold)).mean() - \
                           torch.log(torch.sigmoid(self.threshold - g_neg)).mean()

                    # Manual gradient update
                    layer.zero_grad()
                    loss.backward(retain_graph=True)
                    results["neuron_loss"][i].append(loss.item())
                    with torch.no_grad():
                        for param in layer.parameters():
                            param -= current_lr * param.grad

                # Train readout layer loss
                outputs = self.model(neu_inputs)
                # activations_neu = self.model.forward_activations(neu_inputs)
                # print([torch.max(t).item() for t in activations_neu])
                # outputs = self.model.forward_readout(*activations_neu)
                loss = self.criterion(outputs, labels)
                self.model.readout.zero_grad()
                loss.backward()
                results["readout_loss"].append(loss.item())
                with torch.no_grad():
                    for param in self.model.readout.parameters():
                        param -= current_lr * param.grad

            validation_accuracy = self.validate()
            results["validation_accuracy"].append(validation_accuracy)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Accuracy: {validation_accuracy:.2f}%')
        return results

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                inputs = torch.cat(
                    [images.view(images.size(0), -1), 0.1 * torch.ones((labels.size(0), 10), device=self.device)],
                    dim=1)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        # print(f'Validation Accuracy: {accuracy:.2f}%')
        return accuracy

    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        images_shown = 0
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                inputs = torch.cat([images.view(images.size(0), -1), 0.1 * torch.ones((labels.size(0), 10), device=self.device)], dim=1)
                outputs = self.model(inputs)
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
