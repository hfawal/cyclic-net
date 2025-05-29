import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_dim, common_shape, hidden_shape, output_shape, hidden_size, output_dim = 1, device = 'cpu'):
        super(RNN, self).__init__()
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.device = device
        previous_dim = input_dim + hidden_size
        common_layers = []
        for shape in common_shape:
            common_layers.append(nn.Linear(previous_dim, shape))
            common_layers.append(nn.ReLU())
            previous_dim = shape
        self.common_layers = nn.Sequential(*common_layers)

        common_output = previous_dim       

        hidden_layers = []
        for shape in hidden_shape:
            hidden_layers.append(nn.Linear(previous_dim, shape))
            hidden_layers.append(nn.ReLU())
            previous_dim = shape
        self.hidden_layers = nn.Sequential(*hidden_layers)

        output_layers = []
        previous_dim = common_output
        for shape in output_shape:
            output_layers.append(nn.Linear(previous_dim, shape))
            output_layers.append(nn.ReLU())
            previous_dim = shape
        output_layers.append(nn.Linear(previous_dim, output_dim))
        self.output_layers = nn.Sequential(*output_layers)

    def forward(self, x, h):
        common = self.common_layers(torch.cat((x, h), dim=1))
        h = self.hidden_layers(common)
        x = self.output_layers(common)
        if not self.training:
            x = torch.sigmoid(x)
            x = x.nan_to_num(0.0)
            x = torch.clamp(x, min=0.0, max=1.0)
        return x, h
    
    def init_hidden(self, batch_size):
        return nn.init.kaiming_normal_(torch.empty(batch_size, self.hidden_size, device=self.device))