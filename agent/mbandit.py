import numpy as np


class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon=0.1):
        """
        Initialize the epsilon-greedy multi-armed bandit.

        Parameters:
        - n_arms (int): Number of arms (actions).
        - epsilon (float): Probability of exploration (choose a random arm).
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)      # Count of times each arm has been pulled
        self.values = np.zeros(n_arms)      # Estimated value (mean reward) for each arm

    def select_arm(self):
        """
        Select an arm to pull based on the epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploitation: choose the arm with the highest estimated reward
            return np.argmax(self.values)

    def select_best_arm(self):
        return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        """
        Update the estimated value of the chosen arm after receiving a reward.

        Parameters:
        - chosen_arm (int): The index of the chosen arm.
        - reward (float): The observed reward after choosing the arm.
        """
        # Incremental update to calculate new estimated value
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        # New estimated value using incremental formula
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward


