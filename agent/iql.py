# Implicit Q Learning (IQL)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

EXP_ADV_MAX = 100.

from agent.models.q import TwinQNetwork
from agent.models.value import ValueNetwork
from agent.models.policy import PolicyNetwork


def asymmetric_l2_loss(u, tau):
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class IQL:
    def __init__(self, args):
        self.args = args
        self.gamma = args.agent.gamma
        self.beta = args.agent.beta
        self.device = args.agent.device
        self.batch_size = args.agent.batch_size
        self.tau = args.agent.tau
        self.alpha = args.agent.alpha

        # Initialize networks
        self.q_network = TwinQNetwork(args)
        self.q_target = copy.deepcopy(self.q_network).requires_grad_(False).to(self.device)  # Target Q network
        self.value_network = ValueNetwork(args)
        self.policy_network = PolicyNetwork(args)

        # Initialize optimizers
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=args.agent.lr)
        self.v_optimizer = optim.Adam(self.value_network.parameters(), lr=args.agent.lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=args.agent.lr)

        # Cosine Annealing scheduler for policy network learning rate
        self.policy_lr_schedule = CosineAnnealingLR(self.policy_optimizer, T_max=args.agent.max_steps)

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        with torch.no_grad():
            target_q = self.q_target(state_batch, action_batch)
            next_v_value = self.value_network(next_state_batch)

        # Update value function
        self.v_optimizer.zero_grad()
        v_value = self.value_network(state_batch)
        adv = target_q - v_value
        v_loss = asymmetric_l2_loss(adv, self.tau)
        v_loss.backward()
        self.v_optimizer.step()

        # Update Q function
        self.q_optimizer.zero_grad()
        target = reward_batch + (1 - done_batch) * self.gamma * next_v_value
        qs = self.q_network.both(state_batch, action_batch)
        q_loss = sum(F.mse_loss(q, target) for q in qs) / len(qs)
        q_loss.backward()
        self.q_optimizer.step()

        self.update_target_q_network()

        # policy update
        self.policy_optimizer.zero_grad()
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_action = self.policy_network(state_batch)
        if isinstance(policy_action, torch.distributions.Distribution):
            bc_losses = -policy_action.log_prob(action_batch)
        elif torch.is_tensor(policy_action):
            assert policy_action.shape == action_batch.shape
            bc_losses = torch.sum((policy_action - action_batch)**2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        policy_loss.backward()
        self.policy_optimizer.step()
        self.policy_lr_schedule.step()

        return v_loss.item(), q_loss.item(), policy_loss.item()

    def get_action(self, state):
        return self.policy_network(torch.as_tensor(state, dtype=torch.float32, device=self.device)).detach().cpu().numpy()

    def update_target_q_network(self):
        # Update the target Q network using an exponential moving average (EMA)
        for target_param, param in zip(self.q_target.parameters(), self.q_network.parameters()):
            target_param.data.mul_(1. - self.alpha).add_(param.data, alpha=self.alpha)
            # target_param.data.copy_(self.alpha * param.data + (1 - self.alpha) * target_param.data)

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

            v_loss, q_loss, policy_loss = self.update(state, action, reward, next_state, done)
            # q_loss = self.update_q_network(state, action, reward, next_state, done)
            # v_loss = self.update_value_network(state, action)
            # policy_loss = self.update_policy_network(state, action)

            if (epoch + 1) % 100 == 0:
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

        if self.args.experiment.plot:
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









