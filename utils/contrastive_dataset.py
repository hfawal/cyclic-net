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

        # Set PyTorch seed for consistent random number generation.
        self.generator = torch.Generator().manual_seed(seed)

        # Load all data to determine sizes.
        base_data = list(base_dataset)
        self.length = len(base_data)

        # Get shape of single example.
        example, _ = base_data[0]
        example_shape = example.view(-1).shape if flatten_examples else example.shape
        example_size = example_shape.numel()
        concat_size = example_size + num_classes
        self.original_shape = example_shape

        # Allocate tensors to store the contrastive data.
        self.pos_examples = torch.empty((self.length, concat_size))
        self.neg_examples = torch.empty((self.length, concat_size))
        self.neu_examples = torch.empty((self.length, concat_size))
        self.labels = torch.empty((self.length,), dtype=torch.long)

        # Precompute neutral one-hot vector (all equal probs)
        self.neutral_label = torch.ones(num_classes) / num_classes

        # Fill the allocated tensors.
        for i, (data, label) in enumerate(base_data):

            # Optionally flatten examples.
            data_proc = data.view(-1) if flatten_examples else data

            # One-hot correct label, concatenate to data.
            label_one_hot = funcs.one_hot(torch.tensor(label), num_classes).float()
            self.pos_examples[i] = torch.cat([data_proc, label_one_hot], dim=0)

            # One-hot incorrect label, concatenate to data.
            torch.randint(1, self.num_classes, (1,)).item()
            offset = torch.randint(1, self.num_classes, (1,), generator=self.generator)
            wrong_label = ((label + offset) % self.num_classes).item()
            wrong_label_one_hot = funcs.one_hot(torch.tensor(wrong_label), num_classes).float()
            self.neg_examples[i] = torch.cat([data_proc, wrong_label_one_hot], dim=0)

            # One-hot neutral label, concatenate to data.
            self.neu_examples[i] = torch.cat([data_proc, self.neutral_label], dim=0)

            # Integer label.
            self.labels[i] = label


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Retrieve the data from the stored tensors.
        pos = self.pos_examples[idx]
        neg = self.neg_examples[idx]
        neu = self.neu_examples[idx]
        label = self.labels[idx]

        # Unflatten the examples if necessary.
        if not self.flatten_examples:
            pos_example = pos[:-self.num_classes].reshape(self.original_shape)
            neg_example = neg[:-self.num_classes].reshape(self.original_shape)
            neu_example = neu[:-self.num_classes].reshape(self.original_shape)
            pos = torch.cat([pos_example.view(-1), pos[-self.num_classes:]], dim=0)
            neg = torch.cat([neg_example.view(-1), neg[-self.num_classes:]], dim=0)
            neu = torch.cat([neu_example.view(-1), neu[-self.num_classes:]], dim=0)

        return pos, neg, neu, label