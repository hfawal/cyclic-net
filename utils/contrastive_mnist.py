from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from utils.contrastive_dataset import ContrastiveDataset


class ContrastiveMNISTDataLoader:

    def __init__(self,
                 root: str = './data',
                 download: bool = True,
                 seed: int = 42
                 ):
        """
        Initializes datasets for contrastive learning of MNIST.

        :param root: The directory to look for/save downloaded data in.
        :param download: Whether to download the data if not found.
        :param seed: The seed used for random number generation of negative examples,
        and also for shuffling the datasets upon call to load_data() if necessary.
        """

        self.seed = seed
        self.num_classes = 10

        # Define transformations for the MNIST images.
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to PyTorch tensor.
        ])

        # Download and create the train dataset.
        self.train_dataset = ContrastiveDataset(
            datasets.MNIST(
                root=root,
                train=True,
                download=download,
                transform=self.transform
            ),
            num_classes=self.num_classes,
            seed=self.seed
        )

        # Download and create the test dataset.
        self.test_dataset = ContrastiveDataset(
            datasets.MNIST(
                root=root,
                train=False,
                download=download,
                transform=self.transform
            ),
            num_classes=self.num_classes,
            seed=self.seed
        )

    def load_data(self,
                  val_size: float = 0.1,
                  batch_size: int = 64,
                  num_workers: int = 4,
                  shuffle: bool = True
                  ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Creates training, validation, and testing data loaders for contrastive learning
        on the MNIST dataset.

        :param val_size: The proportion of the training set to be used for validation.
        :param batch_size: The size of batches in the dataloaders.
        :param num_workers: The number of workers (cores) the dataloaders should use.
        :param shuffle: Whether the training set data loader should shuffle the examples.

        :return: Dataloaders for the train set, validation set, and test set in that order.
        These dataloaders return tuples of (positive example, negative example, neutral example,
        label) where the first three tensors are the data concatenated with probability vectors
        corresponding to the given label, an incorrect label, and equal probability per class.
        At inference time you should use the neutral example tensor for prediction.
        """

        # Split train_dataset into train and val sets.
        num_train = len(self.train_dataset)
        num_val = int(val_size * num_train)
        num_train -= num_val
        train_subset, val_subset = random_split(
            self.train_dataset, [num_train, num_val],
            generator=torch.Generator().manual_seed(self.seed)
        )

        # Create DataLoaders.

        train_loader = DataLoader(
            dataset=train_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        val_loader = DataLoader(
            dataset=val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        return train_loader, val_loader, test_loader


