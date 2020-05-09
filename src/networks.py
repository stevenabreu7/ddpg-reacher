import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """ Actor (policy) for RL agent represented by a neural network: state -> action """

    def __init__(self, state_size, action_size, seed, n_hidden_units=[512, 256, 128], lower_init=-3e-3, upper_init=3e-3, batch_norm=True):
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
        self.batch_norm = batch_norm

        self.n_layers = len(n_hidden_units)
        self.state_size = state_size
        self.action_size = action_size

        self.in_layer = nn.Linear(state_size, n_hidden_units[0])
        if self.batch_norm:
            self.in_layer_bn = nn.BatchNorm1d(n_hidden_units[0])
        self.hid_layers = nn.Sequential(*[
            nn.Linear(n_hidden_units[i], n_hidden_units[i+1]) for i in range(self.n_layers - 1)
        ])
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
        # bias
        self.in_layer.bias.data.fill_(0.1)
        for layer in self.hid_layers:
            layer.bias.data.fill_(0.1)
        self.out_layer.bias.data.fill_(0.1)

    def forward(self, state, out_act=torch.tanh):
        """ Forward pass of a state through the network to get an action.

        Params:
            state
            out_act (torch activation function): which activation function to use
                default: use tanh function
        """
        x = self.in_layer(state)
        if self.batch_norm:
            x = self.in_layer_bn(x)
        x = F.relu(x)

        for layer in self.hid_layers:
            x = F.relu(layer(x))
        output = out_act(self.out_layer(x))
        return output


class Critic(nn.Module):
    """ Critic (value) for RL agent represented by a neural network: state -> value (float) """

    def __init__(self, state_size, action_size, seed, n_hidden_units=[512, 256, 128], lower_init=-3e-3, upper_init=3e-3, batch_norm=True):
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
        self.batch_norm = batch_norm

        self.n_layers = len(n_hidden_units)
        self.state_size = state_size
        self.action_size = action_size

        self.in_layer = nn.Linear(state_size, n_hidden_units[0])
        if self.batch_norm:
            self.in_layer_bn = nn.BatchNorm1d(n_hidden_units[0])
        hid_layers = [
            nn.Linear(n_hidden_units[0] + action_size, n_hidden_units[1])
        ]
        hid_layers += [
            nn.Linear(n_hidden_units[i], n_hidden_units[i+1]) for i in range(1, self.n_layers - 1)
        ]
        self.hid_layers = nn.Sequential(*hid_layers)
        self.out_layer = nn.Linear(n_hidden_units[-1], 1)
    
    def reset_parameters(self):
        """ Reset weights to uniform random intialization. Hidden layers according to `hidden_init`
        function. Output layer according to lower and upper bound given by class parameters.
        """
        self.in_layer.weight.data.uniform_(*hidden_init(self.in_layer))
        for layer in self.hid_layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.out_layer.weight.data.uniform_(self.lower_init, self.upper_init)
        # bias
        self.in_layer.bias.data.fill_(0.1)
        for layer in self.hid_layers:
            layer.bias.data.fill_(0.1)
        self.out_layer.bias.data.fill_(0.1)

    def forward(self, state, action):
        """ Forward pass of a state through the network to get a value.
        """
        x = self.in_layer(state)
        if self.batch_norm:
            x = self.in_layer_bn(x)
        x = F.relu(x)
        
        x = torch.cat((x, action.float()), dim=1)
        for layer in self.hid_layers:
            x = F.relu(layer(x))
        output = self.out_layer(x)
        return output
