import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define the DQN model with 1 dense layer
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Single fully connected layer
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x


# Replay buffer class to store experiences
from collections import deque
import random
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# Parameters
# input_size = 12  # 12 features in the input (1x12 numpy array)
# output_size = 4  # Number of actions in the action space (for example)
#
# # Initialize the model
# model = DQN(input_size, output_size)
#
# # Define the optimizer and loss function
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss_fn = nn.MSELoss()
#
# # Example input: numpy array of shape (1, 12)
# state = np.random.rand(1, 12).astype(np.float32)
#
# # Convert the input numpy array to a torch tensor
# state_tensor = torch.from_numpy(state)
#
# # Forward pass through the network
# q_values = model(state_tensor)
#
# # Print the Q-values for each action
# print("Q-values:", q_values)
#
# # Example target Q-values and loss computation (random example)
# target_q_values = torch.rand(1, output_size)  # Mock target values for each action
# loss = loss_fn(q_values, target_q_values)
#
# # Backpropagation
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
#
# print("Updated Q-values:", model(state_tensor))
