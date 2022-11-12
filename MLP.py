import torch as th
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(MLP, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.fcs(x) + self.linear_shortcut(x)