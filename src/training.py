from collections import deque
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys


def run_ddpg_training(env, agent, n_episodes):
    brain_name = env.brain_names[0]
    scores = []
    scores_window = deque(maxlen=100)
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
        np.save("scores.npy", scores)
        # print scores
        print('\repisode {}\t score: {:.4f}\taverage: {:.4f}'.format(
            i_episode, score, np.mean(scores_window)
        ), end="\n" if i_episode % 100 == 0 else "")
        sys.stdout.flush()
        # check if solved
        if len(scores) > 100 and np.mean(scores_window) > 30:
            print("\nsolved environment!")
            agent.save("solved")
            break
    agent.save("end")
    return scores
