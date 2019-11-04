#!/usr/bin/env python
# coding: utf-8
import operator
import abc
from copy import deepcopy
from .policy import *
from collections import deque
from numpy import random as rd

### Learning solver

class ActionValueRL(abc.ABC):
    """ActionValueRL:
        Performs policy optimization using action-value learning
    """
    def __init__(self,
                 Agent=None,
                 Environment=None,
                 params={},
                 ):
        self._Agent = Agent
        self._Environment = Environment
        # not a copy : if Environment._state_space changes (which can happen during exploration)
        # self._state_space should be updated as well
        self._state_space = Environment._state_space

        # Q is a dict with keys {state} and values dict {action: value}
        # To call Q(s,a): self._Q[state][action]
        self._Q = dict(zip(self._Environment._state_space, [{},]*len(self._Environment._state_space)))
        self._max_count = int(1e5) # max number of steps per epoch
        self._iter_per_epoch = []
        self._total_reward_per_epoch = []

        self._policy_params = params.get('policy_params')
        self._agent_params = params.get('agent_params')
        self._exp_replay_params = params.get('exp_replay', {})

        self._learning_rate = params.get('learning_rate')
        self._discount_factor = params.get('discount_factor')

        # set Policy
        self._policy_name = ''
        self._Policy = None
        self.reset(self._policy_params, self._agent_params)

        self._exp_replay_buffer = deque()
        self._do_exp_replay = len(self._exp_replay_params)>0
        self._exp_replay_buffer_size = self._exp_replay_params.get('exp_replay_buffer_size', 0)
        self._exp_replay_batch_size = self._exp_replay_params.get('exp_replay_batch_size', 0)

    def updateQ(self):
        """
        Wrapped in a method so that it can be invoked during learning, to keep self._Q updated
        when self._Environment._state_space changes
        """
        for state in self._state_space:
            if state not in set(self._Q.keys()):
                self._Q[state ] = {}

    def updateExpReplayBuffer(self, transition):
        """
        Add a new transition=(state, action, next_state, reward) to the
        experience replay buffer. Remove older transition if the buffer size
        limit is reached.
        """
        self._exp_replay_buffer.append(transition)
        if len(self._exp_replay_buffer) > self._exp_replay_buffer_size:
            self._exp_replay_buffer.popleft()

    def sampleFromExpReplay(self):
        """
        Sample a transition uniformly from the experience replay buffer.
        Sadly cannot use rd.choice with tuple elements inside the deque.
        """
        idx = rd.randint(len(self._exp_replay_buffer), size=self._exp_replay_batch_size)
        return [self._exp_replay_buffer[i] for i in idx]

    # initialize to Q(s,a)=0
    def reset(self, policy_params={}, agent_params={}):
        state_space = deepcopy(self._state_space)
        for state in state_space:
            actions = self._Environment.actionStateMap(state)
            q_state = {}
            for action in actions:
                q_state[action] = 0
            self._Q[state] = q_state

        # if policy_params is not empty, reset policy as well
        if policy_params:
            self.setPolicy(policy_params)

        # if agent_params is not empty, reset agent as well
        if agent_params:
            self._Agent.reset(agent_params.get('state'))

    def getQ(self, state, action):
        return self._Q[state][action]

    def setQ(self, state, action, new_value):
        self._Q[state][action] = new_value

    def setPolicy(self, policy_params={}):
        self._policy_name = policy_params.get('policy_name')
        if self._policy_name == 'Random':
            self._Policy = RandomPolicy(self._Environment)
        elif self._policy_name == 'Greedy':
            self._Policy = GreedyPolicy(self._Environment)
        elif self._policy_name == 'EpsilonGreedy':
            self._Policy = EpsilonGreedyPolicy(self._Environment, policy_params.get('epsilon'))
        else:
            raise NameError( 'Policy name not found : {}'.format(self._policy_name) )

    # optimal path is the one where each action is selected using the greedy policy
    def optimalPath(self, initial_state):
        count, total_reward = 0, 0
        state = initial_state
        state_history = [initial_state]
        reward_history = [0]
        Greedy = GreedyPolicy(self._Environment)
        while not self._Environment.isTerminal(state) and count < self._max_count:
            #action = max(self._Q[state].items(), key=operator.itemgetter(1))[0]
            action = greedyPickAction(state, self._Q)
            state, reward = self._Environment.getNexStateFromAction(state, action)
            state_history.append(state)
            count += 1
            reward_history.append(reward)

        return state_history, reward_history

    def learn(self, n_epochs, init_state):
        for count_epoch in range(n_epochs):
            # initialize state
            self._Environment.resetAgentState(self._Agent, state=init_state)
            state = self._Agent.state
            self._Environment.actionStateMap(state)

            # choose action from state using policy derived from Q
            action = self._Policy.pickAction(state, {'Q': self._Q})
            count = 0
            total_reward = 0.0

            while not self._Environment.isTerminal(state) and count < self._max_count:
                # take action, observe next_state and reward
                next_state, reward = self._Environment.getNexStateFromAction(state, action)
                transition = (state, action, next_state, reward)

                if self._do_exp_replay:
                    self.updateExpReplayBuffer(transition)
                    replay_transitions = self.sampleFromExpReplay()
                else:
                    replay_transitions = [transition]

                for replay_transition in replay_transitions:
                    self.updateQ()
                    state, action, next_state, reward = replay_transition
                    # choose next_action from next_state using policy derived from Q
                    next_action = self._Policy.pickAction(next_state, {'Q': self._Q})

                    # compute TD target and update Q
                    TD_target = self.TD_update(state, action, reward, next_state, next_action)

                    # update Q
                    new_value = self.getQ(state, action)+self._learning_rate*TD_target
                    self.setQ(state, action, new_value)

                _, _, next_state, reward = transition

                # move to next state, update Agent internal state
                state = next_state
                next_action = self._Policy.pickAction(next_state, {'Q': self._Q})
                action = next_action

                self._Agent.move(next_state, reward)
                count += 1
                total_reward += reward
            self._iter_per_epoch.append(count)
            self._total_reward_per_epoch.append(total_reward)

    @abc.abstractmethod
    def TD_update(self, state, action, reward, next_state, next_action):
        """
        Different TD targets lead to different learning algo : SARSA or Q-learning.
        """
        pass

class Sarsa(ActionValueRL):
    """SARSA:
        SARSA implementation of on-policy TD
    """
    def __init__(self, Agent=None, Environment=None, params={}):
        super().__init__(Agent=Agent, Environment=Environment, params=params)
        self._learning_rate = params.get('learning_rate')
        self._discount_factor = params.get('discount_factor')

    def TD_update(self, state, action, reward, next_state, next_action):
        """
        State. Action. Reward. State. Action.
        """
        return reward+self._discount_factor*self.getQ(next_state, next_action)-self.getQ(state, action)

class QLearning(ActionValueRL):
    """Q-learning:
        Q-learning implementation of off-policy TD -- learn the optimal policy directly while following a different exploration policy
    """
    def __init__(self, Agent=None, Environment=None, params={}):
        super().__init__(Agent=Agent, Environment=Environment, params=params)

    def TD_update(self, state, action, reward, next_state, next_action):
        """
        new_action is not used since Q-learning is off-policy : the optimal action,
        not an action selected by the current policy, is used in the TD target.
        """
        # off-policy learning : take the max of the Q function at the current state, regardless of the action followed by the agent
        Q_max = max(self._Q[next_state].items(), key=operator.itemgetter(1))[1]
        return reward+self._discount_factor*Q_max-self.getQ(state, action)
