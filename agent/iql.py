# Implicit Q Learning (IQL)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agent.models.q import QNetwork
from agent.models.value import ValueNetwork
from agent.models.policy import PolicyNetwork


class IQL:
    def __init__(self, args):
        self.gamma = args.agent.gamma
        self.beta = args.agent.beta
        self.device = args.agent.device
        self.batch_size = args.agent.batch_size

        # Initialize networks
        self.q_network = QNetwork(args)
        self.value_network = ValueNetwork(args)
        self.policy_network = PolicyNetwork(args)

        # Initialize optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=args.agent.lr)
        self.v_optimizer = optim.Adam(self.value_network.parameters(), lr=args.agent.lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=args.agent.lr)

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

        # for testing
        q_losses = []
        v_losses = []
        policy_losses = []

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

            # Store losses for plotting
            q_losses.append(q_loss)
            v_losses.append(v_loss)
            policy_losses.append(policy_loss)

        # for testing
        import matplotlib.pyplot as plt
        # Scale the losses for visibility
        q_losses_scaled = q_losses / (np.max(q_losses) + 1e-8)
        v_losses_scaled = v_losses / (np.max(v_losses) + 1e-8)
        policy_losses_scaled = policy_losses / (np.max(policy_losses) + 1e-8)

        # Plot the scaled losses
        plt.figure(figsize=(10, 6))
        plt.plot(q_losses_scaled, label="Q Loss (scaled)")
        plt.plot(v_losses_scaled, label="V Loss (scaled)")
        plt.plot(policy_losses_scaled, label="Policy Loss (scaled)")
        plt.xlabel("Epoch")
        plt.ylabel("Scaled Loss")
        plt.title("Scaled Loss Values Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()









