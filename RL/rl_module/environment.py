#!/usr/bin/env python
# coding: utf-8
import numpy as np
from numpy import random as rd
import abc

### Generic class for environment

# An environment has :
# 1. a list of state
# 2. a action function that maps each state to the set of possible next states
# 3. a reward function that maps each (action,state) to a reward
# 4. a terminal state
#
# Those are defined here as abstract method to be specified in each environment

class Environment(abc.ABC):
    """
    Environment:
    Contains the state space, the (state,action)->new_state mapping, (state,action,new_state)->reward mapping
    One of the states is the terminal state}
    """
    def __init__(self):
        self._state_space = []

    @abc.abstractmethod
    def actionStateMap(self, state):
        pass

    @abc.abstractmethod
    def getReward(self, state=None, action=None, next_state=None):
        pass

    @abc.abstractmethod
    def isTerminal(self, state=None):
        pass

    def getNexStateFromAction(self, state=None, action=None):
        next_state = self.actionStateMap(state).get(action)
        reward = self.getReward(state, action, next_state)
        return next_state, reward

    def resetAgentState(self, Agent, state=None):
        """
        Random reset, except if a state is specified
        """
        if state:
            Agent.reset(state=state)
        else:
            i = rd.choice(range(len(self._state_space)))
            Agent.reset(state=self._state_space[i])

    def summary(self):
        print('Terminal state: {}'.format(self._terminal_state))
        for state in self._state_space:
            print('{}'.format(state))
            actions = self.actionStateMap(state)
            for action, next_state in actions.items():
                print('\t{} --> {}'.format(action, next_state))


### Generic gridworld

# Gridworld is a generic environment where the state space is chessboard-like

class Gridworld(Environment):
    """
    Chessboard-like environment
    """
    def __init__(self, n_h, n_v, terminal_state=None):
        super().__init__()
        self._n_h = n_h # number of horizontal cells
        self._n_v = n_v # number of vertical cells
        self._terminal_state = terminal_state
        self._directions = {
            'right': [1, 0],
            'left': [-1, 0],
            'up': [0, 1],
            'down': [0, -1],
        }
        # states are tuples (i,j) representing cells of the grid
        self._state_space = [(i,j) for i in range(n_h) for j in range(n_v)]

    def isTerminal(self, state=None):
        return state == self._terminal_state

    def isLeftWall(self, state):
        return state[0] <= 0

    def isRightWall(self, state):
        return state[0] >= self._n_h-1

    def isDownWall(self, state):
        return state[1] <= 0

    def isUpWall(self, state):
        return state[1] >= self._n_v-1

    def printPath(self, state_history, reward_history):
        grid = np.full((self._n_h, self._n_v), np.nan)
        for state, reward in zip(list(state_history),list(reward_history)):
            grid[state] = reward

        np.set_printoptions(nanstr='.')
        # need to flip the h-axis since grid is a matrix
        print(np.flip(grid.T,0))
        print('Total reward : {}'.format(sum(reward_history)))


# In BasicGridworld, you can access all available right/left/up/down squares, with a reward of -1 for each step

class BasicGridworld(Gridworld):
    """
    Simple grid, only up/down/right/left actions
    """
    def __init__(self, n_h, n_v, terminal_state=None):
        super().__init__(n_h, n_v, terminal_state)

    def actionStateMap(self, state):
        directions = self._directions.copy()

        if self.isLeftWall(state):
            del directions['left']

        if self.isRightWall(state):
            del directions['right']

        if self.isUpWall(state):
            del directions['up']

        if self.isDownWall(state):
            del directions['down']

        # lambda to wrap the transition from state using up/down/left/right
        move_lambda = lambda direction: tuple(np.array(state)+np.array(direction))

        # returns list of possible actions, list of possible next_states
        return dict(zip(list(directions.keys()), list(map(move_lambda, list(directions.values())))))

    def getReward(self, state=None, action=None, next_state=None):
        return -1


# In WindyGridworld, there's an upward wind blowing through the middle of the grid

class WindyGridworld(Gridworld):
    """
    Simple grid, the vertical median line has a wind effect that adds an extra up:
    """
    def __init__(self, n_h, n_v, terminal_state=None):
        super().__init__(n_h, n_v, terminal_state)

    def actionStateMap(self, state):
        directions = self._directions.copy()

        if self.isLeftWall(state):
            del directions['left']

        if self.isRightWall(state):
            del directions['right']

        if self.isUpWall(state):
            del directions['up']

        if self.isDownWall(state):
            del directions['down']

        # if state is on the vertical median line of the grid, each move is affected by an extra up (if possible)
        if state[0] == int(self._n_h/2):
            for key, direction in directions.items():
                potential_next_state = tuple(np.array(state)+np.array(direction))
                if not self.isUpWall(potential_next_state):
                    directions[key] = tuple(np.array(direction)+np.array([0, 1]))

        # lambda to wrap the transition from state using up/down/left/right
        move_lambda = lambda direction: tuple(np.array(state)+np.array(direction))

        # returns list of possible actions, list of possible next_states
        return dict(zip(list(directions.keys()), list(map(move_lambda, list(directions.values())))))

    def getReward(self, state=None, action=None, next_state=None):
        return -1


# in CliffGridWorld, there is a cliff at the bottom of the grid : fall and you start over with a massive negative reward!

class CliffGridworld(Gridworld):
    """
    Simple grid, the bottom row of is a cliff that penalizes the agent if it falls over it
    """
    def __init__(self, n_h, n_v, startover_state=None, terminal_state=None):
        super().__init__(n_h, n_v, terminal_state)
        # initial_state is where the agent is brought back if it falls in the cliff
        self._startover_state = startover_state

    def printCliff(self):
        grid = np.full((self._n_h, self._n_v), '.')
        for i in range(1,self._n_h-1):
            grid[(i, 0)] = '*'
        # need to flip the h-axis since grid is a matrix
        print(np.flip(grid.T,0))

    # overload summary to add a print of the cliff
    def summary(self):
        self.printCliff()
        super().summary()

    # if right above the cliff and the action is 'down' or if at the bottom-left corner and the action is 'right'
    def fallsInHole(self, state, action):
        return ((state[1] == 1 and state[0] not in [0, self._n_h-1]) and action == 'down') or (state == (0, 0) and action == 'right')

    def actionStateMap(self, state):
        directions = self._directions.copy()

        if self.isLeftWall(state):
            del directions['left']

        if self.isRightWall(state):
            del directions['right']

        if self.isUpWall(state):
            del directions['up']

        if self.isDownWall(state):
            del directions['down']

        for key, direction in directions.items():
            if self.fallsInHole(state, key):
                directions[key] = tuple(np.array(self._startover_state)-np.array(state))

        # lambda to wrap the transition from state using up/down/left/right
        move_lambda = lambda direction: tuple(np.array(state)+np.array(direction))

        # returns list of possible actions, list of possible next_states
        return dict(zip(list(directions.keys()), list(map(move_lambda, list(directions.values())))))

    def getReward(self, state=None, action=None, next_state=None):
        if self.fallsInHole(state, action):
            return -100
        else:
            return -1
