'''
MonteCarloApproximation.py : Implementation of function approximation based Monte Carlo algorithms

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
from IRL.utils.Helpers import getValueFromDict

class GradientMonteCarloPrediction:
  
  def __init__(self, nStates, nParams, alpha, gamma, approximationFunctionArgs):
    self.name = "Gradient Monte Carlo Prediction"
    self.nStates = nStates
    self.nParams = nParams
    self.alpha = alpha
    self.gamma = gamma
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.returns = {}
    self.visitCounts = np.zeros([self.nStates], dtype=int)
    self.w = np.zeros([self.nParams], dtype=float)
    
  def update(self, episode):
    T = len(episode)-1
    G = 0.0
    for t in range(T-1, -1, -1):
      state = episode[t]["state"]
      reward = episode[t+1]["reward"]
      G = self.gamma * G + reward
      self.visitCounts[state]+=1
      if state not in self.returns.keys():
        self.returns[state] = G
      else:
        self.returns[state] = self.returns[state] + (1.0/self.visitCounts[state]) * (G - self.returns[state])
    for state, G in self.returns.items():
      self.w += self.alpha * (G - self.af(self.w, state, **self.af_kwargs)) * self.afd(self.w, state, **self.af_kwargs)

  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
    
  def getName(self):
    return self.name
    
  def getMu(self):
    return self.visitCounts/np.sum(self.visitCounts)
    
  def reset(self):
    self.returns = {}
    self.visitCounts = np.zeros([self.nStates], dtype=int)
    self.w = np.zeros([self.nParams], dtype=float)
    