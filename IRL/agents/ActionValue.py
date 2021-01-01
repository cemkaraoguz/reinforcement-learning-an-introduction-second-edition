'''
ActionValue.py : Simple action value learners for single state problems

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
from IRL.utils.Policies import DeterministicPolicy
from IRL.utils.Helpers import getValueFromDict

class ActionValueLearner():

  def __init__(self, nStates, nActions, alpha=None, qboost=0.0, actionSelectionMethod="greedy", epsilon=0.0, c=None):
    self.nStates = nStates
    self.nActions = nActions
    self.alpha = alpha
    self.qboost = qboost
    self.actionSelectionMethod = actionSelectionMethod
    self.epsilon = epsilon
    self.c = c
    self.qTable = np.zeros((self.nStates, self.nActions), dtype=float) + self.qboost
    self.nVisits = np.zeros((self.nStates, self.nActions), dtype=int)
    self.policy = DeterministicPolicy(self.nStates, self.nActions, actionSelectionMethod=self.actionSelectionMethod, epsilon=self.epsilon)
    
  def selectAction(self, state, t=None):
    return self.policy.selectAction(state, actionValues=self.qTable, c=self.c, t=t, N=self.nVisits)

  def update(self, state, action, reward):
    self.nVisits[state,action] += 1
    if(self.alpha is None):
      alpha = (1.0/self.nVisits[state,action])
    else:
      alpha = self.alpha
    self.qTable[state,action] += alpha*(reward - self.qTable[state,action])
    self.policy.update(state, np.argmax(self.qTable[state,:]))
    
  def reset(self):
    self.qTable = np.zeros((self.nStates, self.nActions), dtype=float) + self.qboost
    self.nVisits = np.zeros((self.nStates, self.nActions), dtype=int)
    self.policy.reset()