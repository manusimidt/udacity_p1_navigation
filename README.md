[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Links:

- [Navigation.ipynb](./Navigation.ipynb)
- [Report.pdf](./docs/Report.pdf)
- [Saved model weights](checkpoint.pth)

### Introduction

In this project, you will train an agent to navigate (and collect bananas!) in a large, square world.

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.
Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around
agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete
actions are available, corresponding to:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100
consecutive episodes. This means the agent must collect on average at least 13 yellow bananas without running into a
blue one.

### Example of a trained Model

(I trained this one longer than required by project requirements)
![trained agent](docs/assets/trained-agent.gif)

### Getting Started

1. Download the environment from one of the links below. You need only select the environment that matches your
   operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (
      64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

   (_For Windows users_) Check
   out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64)
   if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows
   operating system.

   (_For AWS_) If you'd like to train the agent on AWS (and have
   not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md))
   , then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to
   obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file.

### Installation Instructions

1. Create (and activate) a new environment with Python 3.6.

    - __Linux__ or __Mac__:
   ```bash
   conda create --name drlnd python=3.6
   source activate drlnd
   ```
    - __Windows__:
   ```bash
   conda create --name drlnd python=3.6 
   activate drlnd
   ``` 
2. Install the dependencies:
   ```bash 
   pip install .
   pip install pandas
   ```
   If you get the error message that torch=0.4.0 could not be found, try the following
   ```bash
   conda install pytorch=0.4.0 -c pytorch
   ```

3. Run the `Navigation.py` file or start a jupyter server and run `Navigation.ipynd`

### Instructions

As mentioned in Point 3 in the Installation Instructions you can either run the jupyter notebook `Navigation.ipynd`, or
the python script `Navigation.py`. After initializing the UnityEnvironment, a Agent object instance is created. All
hyperparameters attributable to the Agent can be set over the constructor.

```python
agent = Agent(state_size, action_size, hidden_sizes=[70, 64],
              gamma=0.992, lr=0.0005, tau=0.002,
              buffer_size=100000, batch_size=64, update_rate=10,
              seed=0)
```

After creating the agent instance you can either train the agent by passing it to the
`train_agent()`function, or you can watch the agent interacting with its environment by calling the `watch_agent()`
function. If you want to load the weights for the local network you can also call the `watch_agent_from_pth_file()`
function.

```python
# with this boolean you can decide if you just want to watch an agent or train the agent yourself
watch_only = True
if watch_only:
    watch_agent_from_pth_file(_env, _brain_name, _agent, './checkpoint.pth')
else:
    scores = train_agent(_env, _brain_name, _agent, n_episodes=1000,
                         eps_start=1, eps_decay=0.995, eps_cutoff=420, eps_end=0.01)
    watch_agent(_env, _brain_name, _agent)
    plot_scores(scores=scores)
```

