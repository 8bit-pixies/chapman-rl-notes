"""
Template for implementing QLearner
"""

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
        Qsprime = np.array([(s_prime, a, self.qtable[s_prime, a]) for a in range(self.num_actions)])        
        # select the best action, and the value of Q prime (llast entry)
        best_action = Qsprime[np.argmax(Qsprime[:,2]), 1]
        
        # set the action according to exploration
        if rand.random() < self.exploration:
            action = randaction
        else:
            action = int(best_action)
        
        if r is not None:
            self.qtable[self.s, self.a] = ((1-self.alpha)* self.qtable[self.s, self.a]) + \
                                            self.alpha*(r + self.gamma * max(Qsprime[:,2]))
            # update exploration value. use same discount rate 
            self.exploration = self.gamma * self.exploration
            # update new state and action
            self.s = s_prime
            self.a = action
        
        if self.verbose: print("s: {}, a: {}, r: {}".format(s_prime, action, r))
        return action

