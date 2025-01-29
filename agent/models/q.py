import torch
import torch.nn as nn
from agent.models.normed_linear import NormedLinear


class QNetwork(nn.Module):
    def __init__(self, args):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(args.env.state_dim + args.env.action_dim, args.agent.n_fc_hidden)
        self.layer_norm = nn.LayerNorm(args.agent.n_fc_hidden)
        self.output = NormedLinear(args.agent.n_fc_hidden, 1)  # Final output layer for Q-value

    def forward(self, state, action):
        x = torch.relu(self.fc(torch.cat([state, action], dim=-1)))
        x = self.layer_norm(x)  # Apply layer normalization
        return self.output(x)  # Output shape: [batch_size, 1]


class TwinQNetwork(nn.Module):
    def __init__(self, args):
        super(TwinQNetwork, self).__init__()
        self.q1 = QNetwork(args)
        self.q2 = QNetwork(args)

    def both(self, state, action):
        return self.q1(state, action), self.q2(state, action)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))

