import torch
import torch.nn.functional as funcs
from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):

    def __init__(self,
                 base_dataset: Dataset,
                 num_classes: int,
                 flatten_examples: bool = False,
                 seed: int = 42
                 ):
        """
        Initializes a dataset for contrastive learning based on a base classification dataset.

        :param base_dataset: A classification Dataset that yields (example, label)
        :param num_classes: The number of classes in the dataset.
        :param flatten_examples: Whether the data should be flattened.
        """

        self.num_classes = num_classes
        self.flatten_examples = flatten_examples

        self.data = []

        # Set PyTorch seed for consistent random number generation.
        self.generator = torch.Generator().manual_seed(seed)

        # Precompute neutral one-hot vector (all equal probs)
        self.neutral_label = torch.ones(num_classes) / num_classes

        for image, label in base_dataset:

            # Optionally flatten examples.
            if self.flatten_examples:
                data_proc = image.view(-1)
            else:
                data_proc = image

            # One-hot correct label, concatenate to data.
            label_one_hot = funcs.one_hot(torch.tensor(label), num_classes).float()
            pos_example = torch.cat([data_proc, label_one_hot], dim=0)

            # One-hot incorrect label, concatenate to data.
            torch.randint(1, self.num_classes, (1,)).item()
            offset = torch.randint(1, self.num_classes, (1,), generator=self.generator)
            wrong_label = (label + offset) % self.num_classes
            wrong_label_one_hot = funcs.one_hot(torch.tensor(wrong_label), num_classes).float()
            neg_example = torch.cat([data_proc, wrong_label_one_hot], dim=0)

            # One-hot neutral label, concatenate to data.
            neu_example = torch.cat([data_proc, self.neutral_label], dim=0)

            self.data.append((pos_example, neg_example, neu_example, label))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]