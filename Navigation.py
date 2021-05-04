import unityagents
from unityagents import UnityEnvironment
import numpy as np

"""
The state space is a ndarray of length 37
There are 4 possible actions:
    0 - move forward
    1 - move backward
    2 - turn left
    3 - turn right
    
    
    
Goal: 
    Get an average score of +13 over 100 consecutive episodes
"""


def watch_random_agent(env: UnityEnvironment, brain: unityagents.BrainParameters, brain_name: str) -> int:
    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
    state = env_info.vector_observations[0]  # get the current state
    score = 0  # initialize the score
    while True:
        action = np.random.randint(brain.vector_action_space_size)  # select an action
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        score += reward  # update the score
        state = next_state  # roll over the state to next time step
        if done:  # exit loop if episode finished
            break
    return score


if __name__ == '__main__':
    _env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")

    # get the default brain
    _brain_name = _env.brain_names[0]
    _brain = _env.brains[_brain_name]
    print("Score of random agent {}".format(watch_random_agent(_env, _brain, _brain_name)))

    _env.close()
