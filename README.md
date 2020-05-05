# Unity Reacher - Continuous Control

Training agents using deep reinforcement learning for the reacher environment from the [Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher).

![Trained Agent](./assets/reacher.gif)

In this environment, a double-jointed arm can move to target locations. A positive reward is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

## Learning task

Goal: 
- keep agent's hand in the goal location for as long as possible

State space (state-based):
- `Box(33)` position, rotation, velocity and angular velocities of the arm

Actions: 
- `Box(4)` torque applications to each of the two joints, values between -1 and +1.

Rewards: 
- +0.1 for each time step that the agent's hand is in the goal location

There are two different versions of the environment, detailed in the following.

### Distributed Training

This project contains two separate versions of the Unity environment:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

Note that your project submission need only solve one of the two versions of the environment. 

#### Option 1: Solve the First Version

The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

#### Option 2: Solve the Second Version

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,
- After each episode, the rewards that each agent received are added up (without discounting), to get a score for each agent. This yields 20 (potentially different) scores, of which the average is taken.
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

## Installation

This project is using Python 3.6.3, make sure the following packages are installed:

```bash
pip install numpy matplotlib torch setuptools bleach==1.5.0 unityagents
```

Download the environment from one of the links below:
- **_Version 1: One (1) Agent_**
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

- **_Version 2: Twenty (20) Agents_**
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

Place the downloaded file(s) in the `src/exec/` folder, and unzip (or decompress) the file. Make sure to update the `file_name` parameter in the code when loading the environment:

```python
env = UnityEnvironment(file_name="src/exec/...")
```

## Training instructions

Follow the instructions in `training.ipynb` to see how an agent can be trained.

Coming soon...

## Experiments

Coming soon...

## Further Work

Coming soon...
