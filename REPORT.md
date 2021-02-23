[//]: # (Image References)

[image1]: imgs/agent.gif "Trained Agent"
[algorithm]: imgs/algorithm.png "Algorithm"
[graph]: imgs/graph.png "Graph"
[training]: imgs/training.png "Graph"
[MADDPG]: imgs/MADDPG.png "MADDPG"

# Unity's Reacher Environment Solution with RL

This project is an implementation of Deep Deterministic Policy Gradient Reinforcement learning algorithm to solve Unity's [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.
This report documents my approach, algorithm used and the results obtained in this implementation.

![Trained Agent][image1]

### Problem Statement

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

### Environment

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5. 

### Approach: Deep Deterministic Policy Gradient (DDPG)

The environment was solved using an Actor-Critic implementation of Deep Deterministic Policy Gradient (DDPG) Reinforcement Learning algorithm.

The Algorithm uses an Actor-Critic architecture based on DPG algorithm(Silver et al., 2014)
The actor function uses a neural network and determines the appropiate action for the agent based on the current state. The critic function uses another neural network to calculate the expected return using the current state.The actor function is trained using the advantage function. 

The implemented algorithm as taken from the paper:

![Algorithm][algorithm]

More about the algorithm can be found here: [Continuous control with deep reinforcement learning (arxiv.org)](https://arxiv.org/pdf/1509.02971.pdf)

#### Code Implementation and Model

The project's code is distributed into 3 files:
- `Tennis.ipynb`
  - Import the python packages.
  - Run the environment with Random action Policy.
  - Train 2 agents using DDPG Algorithm using a single actor-critic model.
  - Plot the scores.
- `ddpg_agent.py`
  - Create instance of Actor and Critic Neural Network.
  - Initialize the Replay-Memory Buffer and Noise using Ornstein-Uhlenbeck's process.
  - Train the Neural Network 5 times using random mini-batches from the Memory Buffer every 3 timesteps.
- `model.py`
  - Implements the Actor Neural Network as follows
    - Input Layer of size equal to dimension of states
    - Linear Layer 1 of size 512 units with ReLU activation
    - Batch Normalisation Layer for Layer 1
    - Linear Layer 2 of size 256 units with ReLU activation
    -  Linear Layer 3 of size 256 units with ReLU activation
    - Output Linear Layer 4 of size equal to number of actions with tanh activation
  - Implements the Critic Neural Network as follows
    - Input Layer of size equal to dimension of states
    - Linear Layer 1 of size 510+4(number of discrete actions) units with ReLU activation
    - Batch Normalisation Layer for Layer 1
    - Linear Layer 2 of size 256 units with ReLU activation
    - Linear Layer 3 of size 256 units with ReLU activation
    - Output Linear Layer 4 of size 1

#### Hyperparameters
````
BUFFER_SIZE = int(1e6)    # replay buffer size
BATCH_SIZE = 256          # minibatch size
GAMMA = 0.99              # discount factor
TAU = 1e-3                # for soft update of target parameters
LR_ACTOR = 1e-4           # learning rate of the actor 
LR_CRITIC = 5e-3          # learning rate of the critic
WEIGHT_DECAY = 0          # L2 weight decay
TRAIN_EVERY = 3           # Number of timesteps after which training occurs
TRAN_MINI_BATCHES = 5     # Number of Batches to train every 20 time steps
EPSILON = 1.0             # Noise factor
EPSILON_DECAY = 0.99999   # Noise factor decay
OUNOISE_SIGMA = 0.6       # Value of Sigma while calculating the Ornstein-Uhlenbeck Noise
````
#### Training optimizations:
- Used Batch Size of 256 with Replay buffer of size 1e6 experiences
- Ornstein-Uhlenbeck Noise to action with Noise decay updated every learn step. The noise is of standard normal distribution so that each agent trains differently.
- Used Gradient Clipping to the Critic Network using `torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)`

### Results

The environment was solved in 905 episodes after getting an average score of 0.52 for continous 100 episodes.
The plot of rewards per episode is as follows.

![Graph][graph]
![Traing][training]

### Ideas for future work
The following optimisations can be implemented to improve the performance and accuracy when training simultanously on multiple agents.
- Implementing [MADDPG](https://arxiv.org/pdf/1706.02275.pdf) using multiple actors but same critic network.
    > MADDPG an adaptation of actor-critic methods that considers action policies of other agents and is able to successfully learn policies that require complex multiagent coordination. Additionally, it introduces a training regimen utilizing an ensemble of policies for each agent that leads to more robust multi-agent policies.
    It performes much better when compared to existing methods in cooperative as well as competitive scenarios, where agent populations are able to discover various physical and informational coordination strategies.
    ![MADDPG][MADDPG]

### References
- The Code is based on code implementation of DDPG with OpenAI Gym's BipedalWalker-v2 environment in Udacity's [Deep Reinforcement Learning Nanodegree](https://classroom.udacity.com/nanodegrees/nd893).
- [Continuous control with deep reinforcement learning Paper](https://arxiv.org/pdf/1509.02971.pdf)