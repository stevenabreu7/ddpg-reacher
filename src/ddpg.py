from .networks import Actor, Critic
from .utils import OUNoise, ReplayBuffer
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AgentDDPG:
    """ RL agent trained using Deep Deterministic Policy Gradients. """

    def __init__(self, env, seed, actor_arch=[512, 256, 128], critic_arch=[512, 256, 128], buffer_size=int(1e5), \
                batch_size=128, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.001, noise_mu=0.0, noise_sigma=0.2, \
                noise_theta=0.15, noise_decay=0.99, noise_min_sigma=0.01, weight_decay_critic=0.0, weight_decay_actor=0.0, \
                soft_update_freq=1, hard_update_at_t=1000, gradient_clipping=True):
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
            noise_mu (float): mean for noise process
            noise_sigma (float): standard deviation for noise process
            noise_theta (float): decay/growth parameter for noise process
            noise_decay (float): decay for sigma of noise process (every time agent is reset for new episode)
            noise_min_sigma (float): minimum sigma for noise process (for decaying)
            weight_decay_critic (float): L2 weight decay for critic network
            weight_decay_actor (float): L2 weight decay for actor network
            train_every (int): frequency of training for time steps
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
        self.soft_update_freq = soft_update_freq
        self.hard_update_at_t = hard_update_at_t
        self.gradient_clipping = gradient_clipping
        self.gamma = gamma
        self.tau = tau

        # noise process
        self.noise = OUNoise(self.action_size, self.seed, mu=noise_mu, sigma=noise_sigma, theta=noise_theta, sigma_decay=noise_decay, min_sigma=noise_min_sigma)

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
    
    def episode_step(self):
        """ Reset agent for new episode. Reset noise, update params.
        """
        self.cur_t = 0
        self.noise.reset()
        self.noise.decay_step()
    
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
        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # actor - compute loss
        actions_pred = self.actor_local(states)
        actor_loss = -1.0 * self.critic_local(states, actions_pred).mean()
        # actor - minimize loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # update target networks
        if self.cur_t % self.soft_update_freq == 0:
            self.soft_update(self.critic_local, self.critic_target)
            self.soft_update(self.actor_local, self.actor_target)
        if self.cur_t == self.hard_update_at_t:
            self.hard_update(self.critic_local, self.critic_target)
            self.hard_update(self.actor_local, self.actor_target)
        
    def hard_update(self, local_net, target_net):
        for target_param, source_param in zip(target_net.parameters(), local_net.parameters()):
            target_param.data.copy_(source_param.data)
    
    def soft_update(self, local_net, target_net):
        """ Soft update model parameters, using interpolation parameter tau (class property).

        Params:
            local_net (Torch network): weights to send update
            target_net (Torch network): weights to be updated
        """
        for local_param, target_param in zip(local_net.parameters(), target_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1. - self.tau) * target_param.data)
    
    def save(self, folder):
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
