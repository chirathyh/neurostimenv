import torch
import torch.nn as nn
from agent.models.normed_linear import NormedLinear


class QNetwork(nn.Module):
    def __init__(self, args):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(args.env.state_dim + args.env.action_dim, 16)
        self.layer_norm = nn.LayerNorm(16)
        self.output = NormedLinear(16, 1)  # Final output layer for Q-value

    def forward(self, state, action):
        x = torch.relu(self.fc(torch.cat([state, action], dim=-1)))
        x = self.layer_norm(x)  # Apply layer normalization
        return self.output(x)  # Output shape: [batch_size, 1]
