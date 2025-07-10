import torch
import torch.nn as nn
import torch.optim as optim


class ResidualDynamicsModel(nn.Module):
    def __init__(self, state_dim=48, hidden_dim=128, num_layers=2):
        super().__init__()
        layers = []
        input_dim = state_dim
        # Build hidden MLP
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            input_dim = hidden_dim
        # Final layer maps back to state_dim
        layers.append(nn.Linear(hidden_dim, state_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, s_k):
        """
        s_k: tensor of shape (batch_size, state_dim)
        returns: s_{k+1} prediction, same shape
        """
        delta = self.mlp(s_k)
        return s_k + delta
