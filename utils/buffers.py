import os
from collections import namedtuple, deque
import random
import torch
import h5py
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):

    def __init__(self, args, bufferid, MPI_VAR):
        self.args = args
        self._RANK = MPI_VAR['RANK']
        self.memory = deque([], maxlen=args.replay_buffer_size)
        self.batch_size = args.batch_size

        self.file_path = os.path.join("data", "transitions", f"{bufferid}.h5")
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        # Load the buffer if the file exists, otherwise create an empty file
        if os.path.exists(self.file_path):
            self.load_from_file(self.file_path)
            if self._RANK == 0:
                print(f"Loading buffer from {self.file_path}. Buffer length: {len(self.memory)}")
        else:
            if self._RANK == 0:
                print(f"Creating new buffer file at {self.file_path}...")
            with h5py.File(self.file_path, 'w') as f:
                # Save initial metadata
                meta_group = f.create_group("metadata")
                meta_group.attrs["device"] = self.args.device
                meta_group.attrs["batch_size"] = self.args.batch_size
                meta_group.attrs["replay_buffer_size"] = self.args.replay_buffer_size

    def store(self, *args):
        """Save a transition, convert to tensors"""
        tensor_args = (torch.as_tensor([arg], dtype=torch.float32, device=self.args.device) for arg in args)
        self.memory.append(Transition(*tensor_args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get(self):
        transitions = self.sample(self.batch_size)
        return Transition(*zip(*transitions))

    def close(self):
        self.save_to_file()

    def save_to_file(self, file_path=None):
        """Save the buffer to a file in HDF5 format."""
        file_path = file_path or self.file_path
        with h5py.File(file_path, 'w') as f:
            # Save metadata
            meta_group = f.create_group("metadata")
            meta_group.attrs["device"] = self.args.device
            meta_group.attrs["batch_size"] = self.args.batch_size
            meta_group.attrs["replay_buffer_size"] = self.args.replay_buffer_size

            for idx, transition in enumerate(self.memory):
                group = f.create_group(f"transition_{idx}")
                for field, value in transition._asdict().items():
                    group.create_dataset(field, data=value.cpu().numpy())

    def load_from_file(self, file_path=None):
        """Load the buffer from a file in HDF5 format."""
        file_path = file_path or self.file_path
        with h5py.File(file_path, 'r') as f:
            # Load metadata
            meta_group = f["metadata"]
            saved_replay_buffer_size = meta_group.attrs["replay_buffer_size"]
            # Update replay buffer size to the larger value
            self.args.replay_buffer_size = max(self.args.replay_buffer_size, saved_replay_buffer_size)
            self.memory = deque([], maxlen=self.args.replay_buffer_size)
            if self._RANK == 0:
                print(f"Updated replay_buffer_size to {self.args.replay_buffer_size}.")

            self.memory.clear()
            for transition_name in f.keys():
                if transition_name == "metadata":
                    continue  # Skip metadata
                group = f[transition_name]
                fields = {field: torch.tensor(group[field][()], dtype=torch.float32, device=self.args.device)
                          for field in group.keys()}
                self.memory.append(Transition(**fields))
