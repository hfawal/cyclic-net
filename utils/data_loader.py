import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class MnistDataloader:
    def __init__(self, batch_size=64, shuffle=True, num_workers=2):
        """
        Initialize the MnistDataloader with specified parameters.

        Parameters:
        - batch_size (int): Number of samples per batch.
        - shuffle (bool): Whether to shuffle the dataset at every epoch.
        - num_workers (int): Number of subprocesses to use for data loading.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        # Define transformations for the MNIST images
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to PyTorch tensor
        ])

    def load_data(self, val_size=0.1):
        # Download and create training dataset
        train_dataset = datasets.MNIST(
            root='../../data',
            train=True,
            download=True,
            transform=self.transform
        )

        # Download and create test dataset
        test_dataset = datasets.MNIST(
            root='../../data',
            train=False,
            download=True,
            transform=self.transform
        )

        # Split train_dataset into train and val sets
        num_train = len(train_dataset)
        num_val = int(val_size * num_train)
        num_train = num_train - num_val
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset, [num_train, num_val],
            generator=torch.Generator().manual_seed(42)
        )

        # Create DataLoaders
        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        return train_loader, val_loader, test_loader

