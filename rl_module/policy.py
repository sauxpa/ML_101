#!/usr/bin/env python
# coding: utf-8
from numpy import random as rd
import operator
import abc

### Policy

def greedyPickAction(state, Q):
    """
    Greedy policy:
    for a given state, returns the action that maximizes the current estimate of the Q function
    """
    return max(Q[state].items(), key=operator.itemgetter(1))[0]

class Policy():
    """Policy:
    Contains a state->action mapping
    """
    def __init__(self, Environment):
        self._Environment = Environment

    @abc.abstractmethod
    def pickAction(self, state=None, context={}):
        pass

class EpsilonGreedyPolicy(Policy):
    """Epsilon Greedy policy:
    for a given state, returns the action that maximizes the current estimate of the Q function
    with probability 1-epsilion, otherwise randomly draw an action
    """
    def __init__(self, Environment, epsilon):
        super().__init__(Environment)
        self._epsilon = epsilon

    def pickAction(self, state=None, context={}):
        if rd.uniform() <= 1-self._epsilon:
            return greedyPickAction(state, context.get('Q'))
        else:
            return rd.choice(list(self._Environment.actionStateMap(state).keys()))

class RandomPolicy(EpsilonGreedyPolicy):
    """Random policy:
    for a given state, returns an action drawn uniformly at random
    """
    def __init__(self, Environment):
        super().__init__(Environment, 1)

class GreedyPolicy(EpsilonGreedyPolicy):
    """Greedy policy:
    for a given state, returns the action that maximizes the current estimate of the Q function
    """
    def __init__(self, Environment):
        super().__init__(Environment, 0)
