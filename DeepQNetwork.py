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
        self.model.add(Dense(functools.reduce(lambda x, y: x*y, list(self.num_states)), input_shape=self.num_states, activation='relu'))
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
        else:
            # if reward is nothing set current state to future state...
            self.s = s_prime
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
        
        return action