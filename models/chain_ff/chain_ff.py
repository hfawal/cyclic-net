import torch
import torch.nn as nn


class ChainFF(nn.Module):
    def __init__(self):
        super(ChainFF, self).__init__()
        self.fc1 = nn.Linear(28 * 28 + 10, 2000)
        self.fc2 = nn.Linear(2000, 2000)
        self.fc3 = nn.Linear(2000, 2000)
        self.fc4 = nn.Linear(2000, 2000)
        self.readout = nn.Linear(4 * 2000, 10)

    def forward(self, x):
        a1, a2, a3, a4 = self.forward_activations(x)
        return self.forward_readout(a1, a2, a3, a4)

    def forward_activations(self, x):
        # Starting from flattened data (includes image and labeling data)
        # Dimension: (728 + 10, 1)
        a1 = torch.relu(self.fc1(x))
        a1_norm = a1 / a1.norm(dim=1, keepdim=True)
        a2 = torch.relu(self.fc2(a1_norm.detach()))
        a2_norm = a2 / a2.norm(dim=1, keepdim=True)
        a3 = torch.relu(self.fc3(a2_norm.detach()))
        a3_norm = a3 / a3.norm(dim=1, keepdim=True)
        a4 = torch.relu(self.fc4(a3_norm.detach()))
        a4_norm = a4 / a4.norm(dim=1, keepdim=True)
        return a1_norm, a2_norm, a3_norm, a4_norm

    def forward_readout(self, a1, a2, a3, a4):
        # concat = torch.cat([a2, a3, a4], dim=1)
        concat = torch.cat([a1, a2, a3, a4], dim=1)
        readout = self.readout(concat)
        x = torch.log_softmax(readout, dim=1)  # Log softmax for multi-class classification
        return x
