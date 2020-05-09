from src.ddpg import AgentDDPG
from collections import deque
import json
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys


def run_ddpg_training(env, agentParams, seed, n_episodes, folder):
    print("\n" + folder + "\n")
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(os.path.join(folder, "solved"))
        os.makedirs(os.path.join(folder, "end"))
    # save parameter file
    d = json.dumps(agentParams, indent=2)
    with open(os.path.join(folder, "params.json"), "w") as f:
        f.write(d)
    # create agent
    agent = AgentDDPG(env, seed, **agentParams)
    brain_name = env.brain_names[0]
    scores = []
    scores_window = deque(maxlen=100)
    print('\n...', end="")
    for i_episode in range(1, n_episodes+1):
        # reset agent's noise process
        agent.episode_step()
        # reset environment
        env_info = env.reset(train_mode=True)[brain_name]
        # get current state (for each agent)
        states = env_info.vector_observations
        # initialize score (for each agent)
        score = 0
        for t in range(1000):
            # select action (for each agent)
            actions = agent.act(states)
            # execute actions
            env_info = env.step(actions)[brain_name]
            # get next state, reward, done (for each agent)
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            # learning step for the agent
            agent.step(states[0], actions[0], rewards[0], next_states[0], dones[0])
            # update scores and states (for each agent)
            score += rewards[0]
            states = next_states
            if np.any(dones):
                break
        scores.append(score)
        scores_window.append(score)
        np.save(os.path.join(folder, "scores.npy"), scores)
        with open(os.path.join(folder, "scores.txt"), "a") as f:
            f.write("{:03} {}\n".format(i_episode, score))
        # print scores
        print('\repisode {}\t score: {:.4f}\taverage: {:.4f}'.format(
            i_episode, score, np.mean(scores_window)
        ), end="\n" if i_episode % 100 == 0 else "")
        sys.stdout.flush()

        # save the model every 100 episodes
        if i_episode % 100 == 0:
            agent.save(os.path.join(folder, "params_{}".format(i_episode)))

        # check if solved
        if len(scores) > 100 and np.mean(scores_window) > 30:
            print("\nsolved environment!")
            agent.save(os.path.join(folder, "solved"))
    agent.save(os.path.join(folder, "end"))
    print()
    return scores
