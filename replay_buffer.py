from collections import deque
import random
import numpy as np
import torch

def to_tensor(data, dtype=torch.float32, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    try:
        return torch.tensor(data, dtype=dtype, device=device)
    except ValueError as e:
        print(f"Data conversion error: {e}")
        print(f"Data details: {data}")
        raise

class ReplayBuffer():
    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        """
        Appends a transition to the replay buffer.
        A transition is expected to be a tuple: (state, action, reward, next_state, done)
        """
        if not isinstance(transition, tuple) or len(transition) != 5:
            raise ValueError("Each transition must be a tuple (state, action, reward, next_state, done)")
        self.memory.append(transition)

    def sample(self, sample_size):
        if len(self.memory) < sample_size:
            raise ValueError("Not enough samples in the buffer to draw the requested batch size")

        batch = random.sample(self.memory, sample_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        try:
            states = to_tensor(np.array(states))
            actions = to_tensor(np.array(actions, dtype=np.int64), dtype=torch.int64)  # Explizit als int64
            rewards = to_tensor(np.array(rewards))
            next_states = to_tensor(np.array(next_states))
            dones = to_tensor(np.array(dones), dtype=torch.bool)
        except Exception as e:
            print("Error while processing sample batch:")
            print(f"States shape: {np.array(states).shape}")
            print(f"Actions: {actions}")
            print(f"Rewards shape: {np.array(rewards).shape}")
            print(f"Next States shape: {np.array(next_states).shape}")
            print(f"Dones shape: {np.array(dones).shape}")
            raise e

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
