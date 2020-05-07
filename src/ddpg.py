from collections import namedtuple, deque
from unityagents import UnityEnvironment
import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['figure.figsize'] = 10, 5


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    lim = lim
    return (-lim, lim)


class Actor(nn.Module):
    """ Actor (policy) for RL agent represented by a neural network: state -> action """

    def __init__(self, state_size, action_size, seed, n_hidden_units=[512, 256, 128], lower_init=-3e-5, upper_init=3e-5):
        """ Create a new instance of the actor network.

        Params:
            state_size (int): dimension of the state space
            action_size (int): dimension of the action space
            n_hidden_units (list(int)): number of units in each hidden layer
            lower_init (float): lower bound on random weight initialization in output layer
            upper_init (float): upper bound on random weight initialization in output layer
        """
        super(Actor, self).__init__()
        assert len(n_hidden_units) >= 1

        self.seed = torch.manual_seed(seed)
        self.lower_init = lower_init
        self.upper_init = upper_init

        self.n_layers = len(n_hidden_units)
        self.state_size = state_size
        self.action_size = action_size

        self.in_layer = nn.Linear(state_size, n_hidden_units[0])
        self.hid_layers = [
            nn.Linear(n_hidden_units[i], n_hidden_units[i+1]) for i in range(self.n_layers - 1)
        ]
        self.out_layer = nn.Linear(n_hidden_units[-1], action_size)

        self.reset_parameters()
    
    def reset_parameters(self):
        """ Reset weights to uniform random intialization. Hidden layers according to `hidden_init`
        function. Output layer according to lower and upper bound given by class parameters.
        """
        self.in_layer.weight.data.uniform_(*hidden_init(self.in_layer))
        for layer in self.hid_layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.out_layer.weight.data.uniform_(self.lower_init, self.upper_init)

    def forward(self, state, out_act=F.tanh):
        """ Forward pass of a state through the network to get an action.

        Params:
            state
            out_act (torch activation function): which activation function to use
                default: use tanh function
        """
        x = F.relu(self.in_layer(state))
        for layer in self.hid_layers:
            x = F.relu(layer(x))
        output = out_act(self.out_layer(x))
        return output


class Critic(nn.Module):
    """ Critic (value) for RL agent represented by a neural network: state -> value (float) """

    def __init__(self, state_size, action_size, seed, n_hidden_units=[512, 256, 128], lower_init=-3e-3, upper_init=3e-3):
        """ Create a new instance of the critic network.

        Params:
            state_size (int): dimension of the state space
            n_hidden_units (list(int)): number of units in each hidden layer
            lower_init (float): lower bound on random weight initialization in output layer
            upper_init (float): upper bound on random weight initialization in output layer
        """
        super(Critic, self).__init__()
        assert len(n_hidden_units) >= 2

        self.seed = torch.manual_seed(seed)
        self.lower_init = lower_init
        self.upper_init = upper_init

        self.n_layers = len(n_hidden_units)
        self.state_size = state_size
        self.action_size = action_size

        self.in_layer = nn.Linear(state_size, n_hidden_units[0])
        self.hid_layers = [
            nn.Linear(n_hidden_units[0] + action_size, n_hidden_units[1])
        ]
        self.hid_layers += [
            nn.Linear(n_hidden_units[i], n_hidden_units[i+1]) for i in range(1, self.n_layers - 1)
        ]
        self.out_layer = nn.Linear(n_hidden_units[-1], 1)
    
    def reset_parameters(self):
        """ Reset weights to uniform random intialization. Hidden layers according to `hidden_init`
        function. Output layer according to lower and upper bound given by class parameters.
        """
        self.in_layer.weight.data.uniform_(*hidden_init(self.in_layer))
        for layer in self.hid_layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.out_layer.weight.data.uniform_(self.lower_init, self.upper_init)

    def forward(self, state, action):
        """ Forward pass of a state through the network to get a value.
        """
        x = F.relu(self.in_layer(state))
        
        x = torch.cat((x, action.float()), dim=1)
        for layer in self.hid_layers:
            x = F.relu(layer(x))
        output = self.out_layer(x)
        return output


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., sigma=0.2, theta=0.15):
        """Initialize parameters and noise process.

        Params:
            size (int): dimension of the noise process
            seed (int): random seed
            mu (float): mean of the noise process
            sigma (float): standard deviation of the noise process
            theta (float): decay/growth factor (0 no decay (linear growth) - 1 full decay)
        """
        self.dim = size
        self.seed = random.seed(seed)
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.state = np.ones(self.dim) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.dim) * self.mu

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


class AgentDDPG:
    """ RL agent trained using Deep Deterministic Policy Gradients. """

    def __init__(self, env, seed, actor_arch=[512, 256, 128], critic_arch=[512, 256, 128], buffer_size=int(1e5), \
                batch_size=128, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.001, noise_mu=0.0, noise_sigma=0.2, \
                noise_theta=0.15, weight_decay_critic=0.0, weight_decay_actor=0.0):
        """ Create a new DDPG Agent instance.
        
        Params:
            env: Unity environment for the agent
            actor_arch (list(int)): number of hidden units for each layer in the actor network
            critic_arch (list(int)): number of hidden units for each layer in the critic network
            buffer_size (int): size of the replay buffer
            batch_size (int): number of experiences in each batch
            lr_actor (float): learning rate for Adam optimizer in the actor
            lr_critic (float): learning rate for Adam optimizer in the critic
            gamma (float): discount rate for future rewards
            tau (float): parameter for soft target updates of the networks' weights

            epsilon (float): clipping parameter for clipped surrogate (loss) function
                if none, no clipping is used
            beta (float): regularization parameter (controls exploration)
        """
        # environment
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        self.state_size = self.brain.vector_observation_space_size
        self.action_size = self.brain.vector_action_space_size
        self.seed = seed
        self.cur_t = 0

        # agent hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # noise process
        self.noise = OUNoise(self.action_size, self.seed, mu=noise_mu, sigma=noise_sigma, theta=noise_theta)

        # replay buffer
        self.buffer_size = buffer_size
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size, self.seed)

        # actor and critic network parameters
        self.actor_arch = actor_arch
        self.critic_arch = critic_arch
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.l2_actor = weight_decay_actor
        self.l2_critic = weight_decay_critic

        # actor and critic networks
        self.actor_local = Actor(self.state_size, self.action_size, self.seed, n_hidden_units=self.actor_arch).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.seed, n_hidden_units=self.actor_arch).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=self.lr_actor, weight_decay=self.l2_actor)
        self.critic_local = Critic(self.state_size, self.action_size, self.seed, n_hidden_units=self.actor_arch).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, self.seed, n_hidden_units=self.actor_arch).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=self.l2_critic)
    
    def reset_noise(self):
        """ Reset noise process (start new episode).
        """
        self.noise.reset()
    
    def act(self, state, add_noise=True):
        """ Returns action for given state, following current Actor policy.

        params:
            state: environment state
            add_noise (boolean): whether or not to add noise to the state
        """
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            noise = self.noise.sample()
            action += noise
        return np.clip(action, -1, 1)
    
    def step(self, state, action, reward, next_state, done):
        """ Step with the environment experience, save in memory and learn.
        """
        # save experience to memory buffer
        self.memory.add(state, action, reward, next_state, done)

        # time step
        self.cur_t += 1
        
        if self.cur_t % 10 == 0:
            # learning step (if enough samples)
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def learn(self, experiences):
        """ Update actor and critic network parameters with batch of experience tuples.

        Params:
            experiences (tuple(torch.Tensor)): tuple of (s, a, r, n_s, d) tuples
        """
        states, actions, rewards, next_states, dones = experiences

        # critic - get next q values (from target network)
        next_actions = self.actor_target(next_states)
        next_Q_targets = self.critic_target(next_states, next_actions)
        # critic - compute q targets (current states)
        Q_targets = rewards + (self.gamma * next_Q_targets * (1. - dones))
        # critic - compute loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # critic - minimize loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # actor - compute loss
        actions_pred = self.actor_local(states)
        actor_loss = -1.0 * self.critic_local(states, actions_pred).mean()
        # actor - minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)
    
    def soft_update(self, local_net, target_net):
        """ Soft update model parameters, using interpolation parameter tau (class property).

        Params:
            local_net (Torch network): weights to send update
            target_net (Torch network): weights to be updated
        """
        for local_param, target_param in zip(local_net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1. - self.tau) * target_param.data)
    
    def save(self, folder=None):
        """ Save the current parameters.
        """
        if not folder:
            rng = random.Random()
            folder = "{:x}".format(rng.getrandbits(128))[:10]
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.actor_local.state_dict(), os.path.join(folder, "actor_local.pth"))
        torch.save(self.actor_target.state_dict(), os.path.join(folder, "actor_target.pth"))
        torch.save(self.critic_local.state_dict(), os.path.join(folder, "critic_local.pth"))
        torch.save(self.critic_target.state_dict(), os.path.join(folder, "critic_target.pth"))
