---
## Run 1
One hidden layer
```python 
_agent = Agent(_state_size, _action_size, hidden_sizes=[64], seed=0,
               gamma=0.992, lr=0.005,
               buffer_size=100000, update_rate=10, tau=0.002)
train_agent(_env, _brain_name, _agent, n_episodes=2000,
            eps_decay=0.995)
```

| Episode | Epsilon | Score |
| :---: |:--------:| -----:|
|100|0.576| 0.87|
|300|0.222| 7.69|
|500|0.082|12.92|
|600|0.049|14.91|
|660|0.037|15.99|

---

##Run 2
Two hidden layers
```python 
_agent = Agent(_state_size, _action_size, hidden_sizes=[64,64], seed=0,
               gamma=0.992, lr=0.005,
               buffer_size=100000, update_rate=10, tau=0.002)
train_agent(_env, _brain_name, _agent, n_episodes=2000,
            eps_decay=0.995)
```

| Episode | Epsilon | Score |
| :---: |:--------:| -----:|
|100|0.576| -0.06|
|300|0.222| 3.12|
|500|0.082|9.22|
|600|0.049|10.26|
|700|0.030|10.38|