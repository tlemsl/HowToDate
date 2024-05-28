from collections import defaultdict
import random
import math
import numpy as np

class QLearningAgent:
    def __init__(self, actions, alpha, epsilon, discount):
        self.actions=actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        return self._qvalues[state][action] 

    def set_qvalue(self, state, action, value):
        self._qvalues[state][action] = value


    def get_value(self, state):
        if len(self.actions) == 0:
            return 0.0
        value=max(self.get_qvalue(state, action) for action in self.actions)
        return value

    def update(self, state, action, reward, next_state):
        gamma = self.discount
        learning_rate = self.alpha
        next_value=self.get_value(next_state)
        current_qvalue=self.get_qvalue(state, action)
        updated_qvalue=(1-learning_rate)*current_qvalue+learning_rate*(reward+gamma*next_value)

        self.set_qvalue(state, action, updated_qvalue) 

    def get_best_action(self, state):

        if len(self.actions) == 0:
            return None
        best_action=max(self.actions, key=lambda action: self.get_qvalue(state, action))

        return best_action

    def get_action(self, state):
        action = None
        if len(self.actions) == 0:
            return None
        epsilon = self.epsilon
        if random.random()<epsilon:
          chosen_action=random.choice(self.actions)
        else:
          chosen_action=self.get_best_action(state)

        return chosen_action

