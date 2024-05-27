from collections import defaultdict
import random
import math
import numpy as np


def numpy_to_tuple(array):
    if isinstance(array, np.ndarray):
        return tuple(numpy_to_tuple(sub_array) for sub_array in array)
    else:
        return array


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
        next_array=np.array(next_state)
        s_array=np.array(state)
        t_next_state=numpy_to_tuple(next_array)
        t_state=numpy_to_tuple(s_array)
        next_value=self.get_value(t_next_state)
        current_qvalue=self.get_qvalue(t_state, action)
        updated_qvalue=(1-learning_rate)*current_qvalue+learning_rate*(reward+gamma*next_value)

        self.set_qvalue(t_state, action, updated_qvalue) 

    def get_best_action(self, state):

        if len(self.actions) == 0:
            return None
        best_action=max(self.actions, key=lambda action: self.get_qvalue(state, action))

        return best_action

    def get_action(self, state):
        s_array=np.array(state)
        t_state=numpy_to_tuple(s_array)
        action = None
        if len(self.actions) == 0:
            return None
        epsilon = self.epsilon
        if random.random()<epsilon:
          chosen_action=random.choice(self.actions)
        else:
          chosen_action=self.get_best_action(t_state)

        return chosen_action

