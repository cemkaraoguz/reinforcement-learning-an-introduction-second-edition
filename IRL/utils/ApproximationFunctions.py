'''
ApproximationFunctions.py : Implementations of approximation functions and their derivatives

Cem Karaoguz, 2020
MIT License
'''
import numpy as np
from IRL.utils.Helpers import getValueFromDict
from IRL.utils.Numeric import normalize_softmax

def linearTransform(w, state, action=None, **kwargs):
  ftf = getValueFromDict(kwargs, "ftf")
  return np.dot(w.T, ftf(state, action, **kwargs))

def dLinearTransform(w, state, action=None, **kwargs):
  ftf = getValueFromDict(kwargs, "ftf") 
  return np.array(ftf(state, action, **kwargs))
  
def softmaxLinear(w, state, action=None, **kwargs):
  ftf = getValueFromDict(kwargs, "ftf")
  nActions = getValueFromDict(kwargs, "nActions")
  p = normalize_softmax(np.array([np.dot(w.T, ftf(state, a, **kwargs)) for a in range(nActions)]))
  return p if action is None else p[action]
  
def dLogSoftmaxLinear(w, state, action=None, **kwargs):
  ftf = getValueFromDict(kwargs, "ftf") 
  nActions = getValueFromDict(kwargs, "nActions")
  features = np.array([ftf(state, a, **kwargs) for a in range(nActions)], dtype=float)
  p = softmaxLinear(w, state, **kwargs)
  expectation = np.dot(p, features)
  return features[action] - expectation