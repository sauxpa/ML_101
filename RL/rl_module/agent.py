#!/usr/bin/env python
# coding: utf-8

### Generic class for agent

class Agent():
    """
    Agent:
    Object with a state, the history of past state, and total accumulated reward
    """
    def __init__(self, state=None):
        self._state = state
        self._state_history = [state] if state != None else []
        self._total_reward = 0

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state):
        self._state = new_state
        self._state_history.append(new_state)

    def reset(self, state=None):
        self._state = state
        self.resetHistory()

    def resetHistory(self):
        self._state_history = [];

    def summary(self):
        print('Current state : {}'.format(self._state))
        print('State history : {}'.format(self._state_history))
        print('Total reward  : {}'.format(self._total_reward))

    def move(self, next_state, reward):
        self._state = next_state
        self._state_history.append(next_state)
        self._total_reward += reward
