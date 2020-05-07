from collections import namedtuple, deque
import numpy as np
import random
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., sigma=0.2, theta=0.15, sigma_decay=0.99, min_sigma=0.01):
        """Initialize parameters and noise process.

        Params:
            size (int): dimension of the noise process
            seed (int): random seed
            mu (float): mean of the noise process
            sigma (float): standard deviation of the noise process
            theta (float): decay/growth factor (0 no decay (linear growth) - 1 full decay)
            sigma_decay (float): decay for sigma of noise process (every time agent is reset for new episode)
            _min_sigma (float): minimum sigma for noise process (for decaying)
        """
        self.dim = size
        self.seed = random.seed(seed)
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.sigma_decay = sigma_decay
        self.min_sigma = min_sigma
        self.state = np.ones(self.dim) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.dim) * self.mu
    
    def decay_step(self):
        """ Decay the noise process (sigma). """
        self.sigma = max(self.sigma * self.sigma_decay, self.min_sigma)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """ Fixed-size buffer to store replay experience tuples. """

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params:
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            a_prioritization (float): parameter for prioritization in queue
                0 - no prioritization, 1 - strict prioritization
        """
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory.

        Returns:
            experiences (tuple(s, a, r, s', d)): tuple of lists of states, actions, rewards, next states and done
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
