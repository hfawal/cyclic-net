from utils.ibdm_data_loader import IMDBDataLoader
import torch

class ContrastiveIBDM(IMDBDataLoader):
    def __init__(self, batch_size: int = 64, shuffle: bool = True, num_workers: int = 2):
        super().__init__(batch_size, shuffle, num_workers)

    def collate_batch(self, batch):
        text, labels = super().collate_batch(batch)
        # text is a tensor of shape (batch_size, max_length, embedding_dim)
        # labels is a tensor of shape (batch_size)

        N, L, D = text.shape

        labels = labels.to(torch.int64)

        # Repeat each label L times and reshape to (N, L)
        positive_labels = labels.repeat_interleave(L).reshape(N, L).unsqueeze(2)

        negative_labels  = ~positive_labels.clone()

        neutral_labels = torch.ones_like(positive_labels) * 0.5

        # Create positive and negative pairs
        positive_pairs = torch.cat((text, positive_labels), dim=2)
        negative_pairs = torch.cat((text, negative_labels), dim=2)
        neutral_pairs = torch.cat((text, neutral_labels), dim=2)

        non_zeros = ~(text == 0.0).all(dim=2)
        last_indices = non_zeros.sum(dim=1) - 1
        last_indices = torch.clamp(last_indices, min=0)


        # Create a tensor of shape (3N, L, D+1)
        return positive_pairs, negative_pairs, neutral_pairs, labels, last_indices




