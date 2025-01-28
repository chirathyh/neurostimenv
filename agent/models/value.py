import torch
import torch.nn as nn
from agent.models.normed_linear import NormedLinear


class ValueNetwork(nn.Module):
    def __init__(self, args):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Linear(args.env.state_dim, 16)
        # self.batch_norm = nn.BatchNorm1d(16)  # Batch Normalization
        self.layer_norm = nn.LayerNorm(16)    # Layer Normalization
        self.output = NormedLinear(16, 1)  # Final output layer for V-value

    def forward(self, state):
        x = torch.relu(self.fc(state))  # Apply the linear layer
        # if x.ndim == 2:
        #     x = self.batch_norm(x)  # Apply batch normalization
        x = self.layer_norm(x)  # Apply layer normalization
        return self.output(x)  # Output shape: [batch_size, 1]
