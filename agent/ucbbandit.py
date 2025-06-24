import numpy as np
import math
import pickle


class RobustUCBBandit:
    def __init__(self, n_arms, pretrain=False, checkpoint=None):
        """
        Robust UCB with median-of-means estimator for heavy-tailed rewards.

        Parameters:
        - n_arms (int): Number of arms.
        """
        self.n_arms = n_arms
        self.t = 0  # total time steps
        # For each arm: list of observed rewards
        self.rewards = [[] for _ in range(n_arms)]
        # Count of pulls per arm
        self.counts = np.zeros(n_arms, dtype=int)

        if pretrain:
            self.counts = np.load("../../results/"+checkpoint+"/checkpoints/counts.npy")      # Count of times each arm has been pulled
            self.t = np.load("../../results/"+checkpoint+"/checkpoints/t.npy")

            rewards_path = "../../results/"+checkpoint+"/checkpoints/rewards.pkl"
            with open(rewards_path, "rb") as f:
                loaded_rewards = pickle.load(f)
            if not isinstance(loaded_rewards, list) or len(loaded_rewards) != n_arms:
                raise ValueError(f"Loaded rewards structure invalid: expected list of length {n_arms}")
            # Optionally check lengths match counts
            for i in range(n_arms):
                if len(loaded_rewards[i]) != self.counts[i]:
                    # Warning rather than error: maybe leftover partial state?
                    print(f"Warning: for arm {i}, loaded {len(loaded_rewards[i])} rewards but counts[{i}] = {self.counts[i]}")
            self.rewards = loaded_rewards

            print("Pretrained bandit loaded from checkpoint in experiment: ", checkpoint)
            print(self.counts)
            print(self.rewards)
            print(self.t)

    def select_arm(self):
        """
        Selects an arm according to Robust UCB:
        - For t < n_arms: play each arm once (initialization).
        - Otherwise: compute for each arm i:
            UCB_i = median-of-means estimate + confidence radius,
          and pick argmax UCB_i.
        """
        self.t += 1
        # Initialization: play each arm once
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i

        # Compute UCB indices
        ucb_values = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            n_i = self.counts[i]
            rewards_i = self.rewards[i]
            # Robust estimate: median-of-means if enough pulls, else sample mean
            if n_i >= 2:
                # Choose number of batches B_i
                # Option A: B_i = max(1, floor(log(n_i)))
                B_i = max(1, int(math.floor(math.log(n_i))))
                # Ensure at most n_i batches of size >=1
                B_i = min(B_i, n_i)
                # Batch size
                s_i = n_i // B_i
                if s_i >= 1 and B_i >= 1:
                    # Use the first B_i * s_i rewards for equal-sized batches
                    # (leftover rewards at end are ignored in this batch partition)
                    # For reproducibility/stability, we can shuffle or just use sequential;
                    # here we use the sequential order in which rewards were observed.
                    batch_means = []
                    # Partition into batches
                    for b in range(B_i):
                        start = b * s_i
                        end = start + s_i
                        batch = rewards_i[start:end]
                        # Compute batch mean
                        batch_means.append(np.mean(batch))
                    # Median-of-means
                    est = float(np.median(batch_means))
                else:
                    # Fallback if somehow s_i < 1
                    est = float(np.mean(rewards_i))
            else:
                # n_i == 1: sample mean is just the single reward
                est = float(self.rewards[i][0])

            # Confidence radius: simple form sqrt(2 log t / n_i)
            # Avoid division by zero since counts[i]>0 here
            radius = math.sqrt((2.0 * np.log(max(self.t, 2))) / n_i)
            ucb_values[i] = est + radius

        # Break ties arbitrarily: np.argmax picks first max
        chosen = int(np.argmax(ucb_values))
        return chosen

    def update(self, chosen_arm, reward):
        """
        Update statistics after pulling `chosen_arm` and observing `reward`.
        """
        # Record reward
        self.counts[chosen_arm] += 1
        self.rewards[chosen_arm].append(float(reward))
        # No other state to update immediately; estimates are computed on-the-fly in select_arm()

    def select_best_arm(self):
        """
        Return the arm with highest current median-of-means estimate (without confidence radius).
        Useful after sufficient pulls to identify best arm.
        """
        best_estimates = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            n_i = self.counts[i]
            if n_i == 0:
                best_estimates[i] = float('-inf')
                continue
            rewards_i = self.rewards[i]
            if n_i >= 2:
                B_i = max(1, int(math.floor(math.log(n_i))))
                B_i = min(B_i, n_i)
                s_i = n_i // B_i
                if s_i >= 1 and B_i >= 1:
                    batch_means = []
                    for b in range(B_i):
                        start = b * s_i
                        end = start + s_i
                        batch = rewards_i[start:end]
                        batch_means.append(np.mean(batch))
                    est = float(np.median(batch_means))
                else:
                    est = float(np.mean(rewards_i))
            else:
                est = float(rewards_i[0])
            best_estimates[i] = est
        return int(np.argmax(best_estimates))
