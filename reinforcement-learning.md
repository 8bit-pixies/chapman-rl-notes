Introduction to Reinforcement Learning
======================================

Reinforcement is large topic which deserves treatment similar to the other two branches of machine learning, supervised and unsupervised learning. In this section we will provide motivation and introduction to reinforcement learning, firstly by introducing Q-Learning and then brief exploring how this can be extended using neural networks. 

As always the purpose of this section is not to provide a comprehensive overview, but simply to provide high level introduction to motivate others to dive deeper into reinforcement learning.

What is Reinforcement Learning
------------------------------

Reinforcement learning has many names. It is referred to in literature as:

*  Building _software agents_
*  AI based _decision making_
*  Allowing Robots to _autonomously_ learn through trial and error

As such it is important to differentiate what is and isn't reinforcement learning.

![Roomba](08-rl-decision-making/roomba.jpg)

Consider the roomba robotic vacuum cleaner. It is a robotic vacuum cleaner which randomly moves around the room to clean it. This is _not_ an example of reinforcement learning, as there is no learning involved. However if the roomba, through its sensors slowly and over time began to understand its environment and cleaned the room more and more efficiently; then it would fall under reinforcement learning.

### How is Reinforcement Learning different from Supervised or Unsupervised Learning?

To understand the differences between different types of machine learning, consider the goals of each branch of machine learning. All areas are focused on finding some function/mapping $f$

1.  Supervised Learning: $y=f(x)$
2.  Unsupervised Learning: $f(x)$
3.  Reinforcement Learning: $y \stackrel{\text{z}}{=} f(x)$ 

**Supervised Learning**

For supervised learning, $x$ is your input data, $y$ is your labels and $f$ is the function approximation of $x$ which is typically some kind of model, like decision trees or regression 

**Unsupervised Learning**

The goal within unsupervised learning is typically to learn some kind of efficient representation of the input data. This could be in terms of reconstruction error leading to techniques like PCA or SVD or clustering like k-means or finding latent variables like latent dirichlet allocation. 

**Reinforcement Learning**

Unlike supervised learning, the goal of reinforcement learning is not to directly learn about $y$, rather it is to find the optimal policy of the environment. The input $x$ is normally the "state" and "action" pair, whilst $z$ is typically the reward which is external to the model $f$ being learned. These terms will be explained in more detail shortly.

Q-Learning
----------

Broadly speaking there are two ways of solving reinforcement learning problems. One way is through genetic programming, and the other is through markov decision processes (MDP). In this section we will only be considering MDP. All MDP have four components:

*  States: agent must know about its environment
*  Action: agent must know what it is allowed to do
*  Rewards: we want to reward "good" actions and punish "bad" actions
*  Discount rate: we prefer if we "good" actions come sooner, rather than later

There are many ways to solve MDPs, we will only consider Q-learning as it is one of the more straight-forward models. 

Q-learning aims to directly modelling the action-value function (q-function), with the goal of finding the best policy (or strategy) by selecting the actio which maximizing the value at each state. 

In plain english this q-function is as follows:

>  Q(state, action) = immediate reward + expected future rewards

Slightly more formally, if $s$ is the state, $a$ is action, $r$ is reward, $\gamma$ is discount rate, and $s', a'$ refer to the next state/action respectively (n.b. $r$ in most texts is a function, and has been simplified in these notes),

$$Q(s,a) = r + \gamma \max_{a'}  Q(s', a')$$

We can see from the equation above, that our expected reward for taking action $a$ is the immediate reward plus the discount future rewards if we take the optimal action in the next state.

To "solve" this iterative looking equation, we can use an update rule which is shown below:

$$Q(s, a) \leftarrow Q(s, a) + \alpha(r + \gamma \max_{a'} Q(s', a') - Q(s, a))$$

Whenever a new state action combination is realised we can update the knowledge within this function. 

### Q-learning Pseudo-code

------------------------------------------------------------------------------------------------------------------------
_Initialise variables_

1.  Initialise a "Q learning table" which is the q function value for all state, action pairs (i.e. if we have 10 states, 4 actions, we will have a 10 by 4 table) - set all values to 0. 
2.  Intiailise an exploration rate, we will randomly explore (outside of optimal action) when our random number is lower than exploration rate. As we take more actions, we will lower the exploration rate. 

_Algorithm_

1. Based on state, reward values, update qtable using

$$Q(s, a) \leftarrow Q(s, a) + \alpha(r + \gamma \max_{a'} Q(s', a') - Q(s, a))$$

2. Return action based on optimal action from Qtable or random action if random number generated is less than exploration rate.
------------------------------------------------------------------------------------------------------------------------


```py
import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, 
        num_states=100, 
        num_actions = 4, 
        alpha = 0.2, # learning rate
        gamma = 0.9, # discount
        exploration = 0.99, # exploration starting value
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        
        # qtable is size num_states by num_actions
        self.qtable = np.zeros((self.num_states, self.num_actions))
        
        # initialise some values for initial state
        self.s = 0
        self.a = 0
    
    def query(self, s_prime, r=None):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
                              
        randaction = rand.randint(0, self.num_actions-1)  
        # add to the qtable, the state and reward
        Qsprime = np.array([(s_prime, a, self.qtable[s_prime, a]) 
                             for a in range(self.num_actions)])        
        # select the best action, and the value of Q prime (last entry)
        best_action = Qsprime[np.argmax(Qsprime[:,2]), 1]
        
        # set the action according to exploration
        if rand.random() < self.exploration:
            action = randaction
        else:
            action = int(best_action)
        
        if r is not None:
            # this is the update step which shows the iterative update approach
            self.qtable[self.s, self.a] = ((1-self.alpha)* self.qtable[self.s, self.a]) + \
                                            self.alpha*(r + self.gamma * max(Qsprime[:,2]))
            # update exploration value. use same discount rate 
            self.exploration = self.gamma * self.exploration
            # update new state and action
            self.s = s_prime
            self.a = action
        
        if self.verbose: print("s: {}, a: {}, r: {}".format(s_prime, action, r))
        return action
```

To run the full example please consult the appendix.

Deep Q-Networks
---------------

Deep Q-Networks uses neural networks to approximate q-function. It uses the [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem) which suggests that a sufficiently complex neural network can approximate any function. 

For "simple" problems without obvious relationships between states, this may result in overly complicated neural networks, however it will perform well when the actual rules are ill-defined, or in particular when states and actions are not necessarily discrete. 

To convert the reinforcement learning problem to something which can be solved in a neural network, we will reframe the problem in terms of a loss function:

$$L = \frac{1}{2}(r + \max_{a'} Q(s', a') - Q(s, a))^2$$

Where $r + max_{a'} Q(s', a')$ is our target and $Q(s, a)$ is the prediction. Which means this is analogous to mean squared error:

$$ L = (y - \hat{y})^2$$

With this knowledge we can create some pseudo-code!

-----------------------------------------------------------------------------------------------
1.  Based on the current model; calculate $Q(s, a)$ for all actions
2.  Then calculate the next state's associated values $r + max_{a'} Q(s', a')$
3.  Set the target for calculated action in step 2 to be the target for neural network, 
    everything else is the target as calculated in step 1
4.  Update weights for neural network! (backprop)
-----------------------------------------------------------------------------------------------

### Experience Replay

One important aspect of Deep Q-Networks is the usage of "experience replay". In order to update our neural network, the more effective way is to store the actions and rewards in memory, so that they can be recalled for random mini-batch updating. This can also be used to "inject" human gameplay to train models in this way as well, without having the agent "randomly" explore the space which is particularly important when exploration is potentially expensive. 

```py
import random as rand
import numpy as np
import functools
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape, Merge

class DeepQN(object):

    def __init__(self, \
        num_states=(10,10,), \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        exploration = 0.99, \
        memory = 100, \
        memory_decay = 0.9,
        hidden_size = 100,
        batch_size = 50,
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        
        # keep track of the following stuff
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.exploration = exploration
        self.memory = memory # keep the last X states
        self.memory_dict = [] # list of s, a, r, s'
        
        # this is our current state (s) and action (a)
        # in this framework, s is an array
        self.s = 0
        self.a = 0
        self.hidden_size = hidden_size # this is to put a hidden layer...if needed?
        self.batch_size = batch_size
        
        # this is our neural net which will approximate things...
        self.model = Sequential()
        self.model.add(Dense(functools.reduce(lambda x, y: x*y, list(self.num_states)), 
                             input_shape=self.num_states, activation='relu'))        
        self.model.add(Flatten())
        self.model.add(Dense(self.hidden_size, activation='relu'))
        self.model.add(Dense(self.num_actions, activation='relu'))
        self.model.compile(optimizer='sgd', loss='mse')

    def query(self, s_prime, r=None):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        
        # if r is None then we are not aiming to use this example for training!
        if r is not None:
            # first save in memory
            self.memory_dict.append({'s': self.s, 'a': self.a, 'r': r, 's_prime': s_prime})
            self.memory_dict = self.memory_dict[-self.memory:]
            
            # this step is "hallucinating" game states to help train
            # based on previously seen rewards
            # this is also called 
            # "experience replay"
            # we will use this to transform data to create a batch training dataset
            # now transform to create training batch...
            input_data = np.zeros(tuple([len(self.memory_dict)]+list(self.num_states)))
            targets = np.zeros((len(self.memory_dict), self.num_actions))
            
            # randomly select to do batch training on this...
            for idx in np.random.randint(0, high=len(self.memory_dict), size=self.batch_size):
                sample = self.memory_dict[idx]
                input_data[idx] = sample['s']
                targets[idx, :] = self.model.predict(sample['s'])[0]
                
                Qsa = np.max(self.model.predict(sample['s']))
                targets[idx, sample['a']] = sample['r'] + self.gamma * Qsa
            
            # batch training...
            # input is the states, targets is the rewards
            self.model.train_on_batch(input_data, targets)
        
        # now determine next action to take
        randaction = rand.randint(0, self.num_actions-1)
        
        reward_actions = self.model.predict(self.s) # output of this is 2d array
        # which is why it is index by 0 here. 
        best_action = np.argmax(reward_actions[0])
        
        # set the action according to exploration
        if rand.random() < self.exploration:
            action = randaction
        else:
            action = int(best_action)
            
        # update exploration value. 
        self.exploration = self.exploration * self.gamma
        
        # update new state and action
        self.s = s_prime
        self.a = action
        
        if self.verbose: print("s: {}, a: {}, r: {}".format(np.where(s_prime)[1][0], action, r))
        return action
```

To run the full example please consult the appendix.

Practical applications
----------------------

*  [Parameter optimization](https://gym.openai.com/envs#parameter_tuning)
*  Customer Journey models, [customer call center](http://rankminer.com/reinforcement-learning-big-data-in-call-centers/)
*  High frequency trading models
*  Multi-arm bandit problems; generalising AB testing (e.g. for selecting "best" models in production)



