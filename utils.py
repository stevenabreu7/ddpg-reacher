from collections import deque
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys


def show_scores_plot(scores, folder, save_np=True, save_plot=True):
    """Show the scores plot and optionally save the plot.
        
    Params:
        filename (str): optional filename to save the plot
    """
    if save_np:
        np.save(os.path.join(folder, "scores.npy"), scores)

    window_size = 100
    scores = np.array([np.mean(scores[max(0, idx-window_size+1):idx+1]) for idx in range(len(scores))])

    # plot agent scores
    plt.plot(np.arange(1, 101), scores[:100], linestyle="dashed", color="blue")
    plt.plot(np.arange(101, len(scores)+1), scores[100:], color="blue", label="agent")

    # plot agent scores that solved environment
    win_scores = scores.copy()
    win_scores[win_scores < 13] = np.nan
    plt.plot(np.arange(1, len(scores)+1), win_scores, color="orange", label="solved")

    # marker for target score (solution)
    plt.hlines(30, 1, len(scores), colors=["red"], linestyles=["dashed"], label="goal")

    # marker for episode when solution was first found
    ep_solve = np.argmax(np.array(scores) > 13)
    ep_solve = ep_solve
    if ep_solve:
        plt.vlines(ep_solve, 0, max(max(scores), 13), colors=["green"], linestyles=["dashed"], label="solution")
        plt.annotate("{}".format(ep_solve), (ep_solve - len(scores) / 15, 4.0))

    # labels and legend
    plt.ylabel("average score (over 100 episodes)")
    plt.xlabel("episode")
    plt.legend(bbox_to_anchor=(1.17, 1))

    if save_plot:
        filename = os.path.join(folder, "score_plot.png")
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
