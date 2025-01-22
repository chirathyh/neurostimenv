# Implicit Q Learning (IQL)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class IQL:
    def __init__(self, args):
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.gamma = args.gamma
        self.beta = args.beta
        self.device = args.device
        self.batch_size = args.batch_size

        # Initialize networks
        self.q_network = QNetwork(args.state_dim, args.action_dim)
        self.value_network = ValueNetwork(args.state_dim)
        self.policy_network = PolicyNetwork(args.state_dim, args.action_dim)

        # Initialize optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=args.lr)
        self.v_optimizer = optim.Adam(self.value_network.parameters(), lr=args.lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=args.lr)

        # Loss function
        self.mse_loss = nn.MSELoss()

    def update_q_network(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        self.q_optimizer.zero_grad()
        q_value = self.q_network(state_batch, action_batch)
        next_v_value = self.value_network(next_state_batch).detach()
        target_q = reward_batch + (1 - done_batch) * self.gamma * next_v_value
        q_loss = self.mse_loss(q_value, target_q)
        q_loss.backward()
        self.q_optimizer.step()
        return q_loss.item()

    def update_value_network(self, state_batch, action_batch):
        self.v_optimizer.zero_grad()
        v_value = self.value_network(state_batch)
        q_value = self.q_network(state_batch, action_batch).detach()
        exp_advantages = torch.exp((q_value - v_value) / self.beta)
        weighted_values = exp_advantages * q_value
        v_loss = self.mse_loss(v_value, weighted_values.mean(dim=0, keepdim=True))
        v_loss.backward()
        self.v_optimizer.step()
        return v_loss.item()

    def update_policy_network(self, state_batch, action_batch):
        self.policy_optimizer.zero_grad()
        q_value = self.q_network(state_batch, action_batch).detach()
        v_value = self.value_network(state_batch).detach()
        advantage = q_value - v_value
        weights = torch.exp(advantage / self.beta).clamp(max=100)  # Advantage weighting
        policy_action = self.policy_network(state_batch)
        policy_loss = ((policy_action - action_batch) ** 2 * weights).mean()
        policy_loss.backward()
        self.policy_optimizer.step()
        return policy_loss.item()

    def get_action(self, state):
        return self.policy_network(torch.as_tensor(state, dtype=torch.float32, device=self.device)).detach().cpu().numpy()

    def train(self, replay_memory, epochs=100):
        for epoch in range(epochs):
            if len(replay_memory) < replay_memory.batch_size:
                continue  # Skip if not enough samples in memory

            # Sample batch from replay memory
            batch = replay_memory.get()
            state = torch.cat(batch.state).to(self.device)
            action = torch.cat(batch.action).to(self.device)
            reward = torch.cat(batch.reward).view(-1, 1).to(self.device)
            next_state = torch.cat(batch.next_state).to(self.device)
            done = torch.cat(batch.done).view(-1, 1).to(self.device)

            q_loss = self.update_q_network(state, action, reward, next_state, done)
            v_loss = self.update_value_network(state, action)
            policy_loss = self.update_policy_network(state, action)

            #if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Q Loss = {q_loss:.4f}, V Loss = {v_loss:.4f}, Policy Loss = {policy_loss:.4f}")


# Neural Network Definitions
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(state_dim + action_dim, 16)
        self.output = nn.Linear(16, 1)  # Final output layer for Q-value

    def forward(self, state, action):
        x = torch.relu(self.fc(torch.cat([state, action], dim=-1)))
        return self.output(x)  # Output shape: [batch_size, 1]


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, 16)
        self.output = nn.Linear(16, 1)  # Final output layer for V-value

    def forward(self, state):
        x = torch.relu(self.fc(state))
        return self.output(x)  # Output shape: [batch_size, 1]


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(state_dim, 16)
        self.output = nn.Linear(16, action_dim)  # Match action_dim

    def forward(self, state):
        x = torch.relu(self.fc(state))
        return self.output(x)  # Shape: [batch_size, action_dim]
