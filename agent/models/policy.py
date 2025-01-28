import torch
import torch.nn as nn
from agent.models.normed_linear import NormedLinear


class PolicyNetwork(nn.Module):
    def __init__(self, args):
        super(PolicyNetwork, self).__init__()
        self.args = args
        self.fc = nn.Linear(args.env.state_dim, 16)
        self.layer_norm = nn.LayerNorm(16)
        self.output = NormedLinear(16, args.env.action_dim)  # Match action_dim

    def forward(self, state):
        x = torch.relu(self.fc(state))
        x = self.layer_norm(x)
        actions = torch.sigmoid(self.output(x))  # Shape: [batch_size, action_dim]
        # Split actions
        action1, action2 = actions[:, 0], actions[:, 1]

        # Apply squashing and rescaling
        action1 = action1 * (self.args.env.stimAmplitude_max - self.args.env.stimAmplitude_min) + self.args.env.stimAmplitude_min  # Scale to [1e-3, 10e-3]
        action2 = action2 * (self.args.env.stimFreq_max - self.args.env.stimFreq_min) + self.args.env.stimFreq_min     # Scale to [1, 20]

        # Combine and return
        return torch.stack([action1, action2], dim=1)
