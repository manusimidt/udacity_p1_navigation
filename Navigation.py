from collections import deque

import torch
import unityagents
from unityagents import UnityEnvironment
import numpy as np

from agent import Agent

"""
The state space is a ndarray of length 37
There are 4 possible actions:
    0 - move forward
    1 - move backward
    2 - turn left
    3 - turn right
    
    
    
Goal: 
    Get an average score of +13 over 100 consecutive episodes
    Achieve this in less than 1800 episodes
"""


def watch_agent(env: UnityEnvironment, brain_name: str, agent: Agent) -> None:
    """
    Shows agent simulation
    :param env:
    :param brain_name:
    :param agent:
    :return:
    """
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score

    while True:
        action = agent.act(state, epsilon=0)
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break
    print(f"Agent achieved a score of {score}")


def train_agent(env: UnityEnvironment, brain_name: str, agent: Agent, n_episodes: int,
                eps_start=1.0, eps_end=0.01, eps_decay=0.995) -> []:
    """
    Trans the agent for n episodes
    :param env:
    :param brain_name:
    :param agent:
    :param n_episodes: number of episodes to train
    :param eps_start: epsilon start value
    :param eps_end: epsilon decay per episode
    :param eps_decay: minimum value for epsilon (never stop exploring)
    :return:
    """
    scores: [int] = []
    eps = eps_start
    # store the last 100 scores into a queue to check if the agent reached the goal
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        # reset the environment
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        state = env_info.vector_observations[0]
        score = 0
        while True:
            action: int = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            state = next_state

            agent.step(state, action, reward, next_state, done)

            if done:
                break

        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        if i_episode % 10 == 0:
            print(f"""Episode {i_episode}:
            Epsilon: {eps:.3f}
            Average Score: {np.mean(scores_window):.2f}
            Weights: {agent.target_network.fc1.weight}
            """)

        if np.mean(scores_window) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.local_network.state_dict(), 'checkpoint.pth')
            break
    return scores


if __name__ == '__main__':
    _env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

    # get the default brain
    _brain_name: str = _env.brain_names[0]
    _brain: unityagents.BrainParameters = _env.brains[_brain_name]

    _action_size: int = 4
    _state_size: int = 37

    # print("Score of random agent {}".format(watch_random_agent(_env, _brain, _brain_name)))

    _agent = Agent(_state_size, _action_size, seed=0, update_rate=10, tau=0.002, gamma=0.992, lr=0.001)
    # watch_agent(_env, _brain_name, _agent)
    train_agent(_env, _brain_name, _agent, n_episodes=2000, eps_decay=0.996)
    watch_agent(_env, _brain_name, _agent)

    _env.close()
