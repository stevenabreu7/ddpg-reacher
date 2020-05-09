from src.training import run_ddpg_training
from utils import show_scores_plot
from unityagents import UnityEnvironment
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
import sys

env = UnityEnvironment(file_name="src/exec/Reacher1.app")
seed = 123
n_episodes = 1000

agentParams = {
    "actor_arch": [128, 128],
    "critic_arch": [128, 128],
    "buffer_size": int(1e5),
    "batch_size": 128,
    "lr_actor": 1e-4,
    "lr_critic": 1e-3,
    "gamma": 0.99,
    "tau": 1e-3,
    "noise_mu": 0.0,
    "noise_sigma": 0.2,
    "noise_decay": 1.0,
    "noise_min_sigma": 0.01,
    "noise_theta": 0.15,
    "weight_decay_critic": 0.0,
    "weight_decay_actor": 0.0,
    "soft_update_freq": 1,
    "hard_update_at_t": -1,
    "gradient_clipping": False
}
folder = "06_given_longer"
scores = run_ddpg_training(env, agentParams, seed, n_episodes, folder)
