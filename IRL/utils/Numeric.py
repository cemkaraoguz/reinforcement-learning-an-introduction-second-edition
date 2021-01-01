'''
Numeric.py : implementations of various numeric helper functions

Cem Karaoguz, 2020
MIT License
'''

import numpy as np

def argmax(x):
  return np.random.choice(np.nonzero(x==np.max(x))[0])
  
def normalize_sum(x, **kwargs):
  x_norm = x + np.min(x)
  if np.isclose(np.sum(x_norm),0):
    x_norm = x_norm + (1.0/len(x))
  else:
    x_norm = x_norm/np.sum(x_norm)
  return x_norm
  
def normalize_softmax_nonsafe(x, **kwargs):
  return np.exp(x) / np.sum(np.exp(x))
  
def normalize_softmax(x, **kwargs):
  shiftx = x - np.max(x)
  exps = np.exp(shiftx)
  return exps / np.sum(exps)
  
def normalize_greedy(x, **kwargs):
  argmax_function=kwargs["argmaxfun"]
  x_norm = np.zeros_like(x)
  x_norm[argmax_function(x)] = 1.0
  return x_norm

def normalize_esoft(x, **kwargs):
  argmax_function = kwargs["argmaxfun"]
  epsilon = kwargs["epsilon"]
  x_norm = np.zeros_like(x) + epsilon/(len(x) - 1)
  x_norm[argmax_function(x)] = 1.0 - epsilon
  return x_norm

def mapValues(val, minSrc, maxSrc, minDest, maxDest):
  aux = (val - minSrc)/(maxSrc-minSrc)
  return aux*(maxDest-minDest) + minDest