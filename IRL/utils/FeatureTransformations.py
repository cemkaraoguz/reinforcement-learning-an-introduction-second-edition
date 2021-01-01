'''
FeatureTransformations.py : Implementations of feature transformation methods presented in the book

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
from IRL.utils.Helpers import getValueFromDict
from IRL.utils.Numeric import mapValues
  
def buildStateVector(state):
  if np.isscalar(state):
    stateVector = np.array([state])
  else:
    stateVector = np.array(state)
  return stateVector
  
def stateAggregation(state, action=None, **kwargs):
  '''
  Simple tile coding with one grid tiling and one state dimension
  '''
  nParams = getValueFromDict(kwargs, "nParams") 
  nStates = getValueFromDict(kwargs, "nStates") 
  if action is None:
    nActions = 1
    idx_action = 0
  else:
    nActions = getValueFromDict(kwargs, "nActions")
    idx_action = action
  stateFeatureVectorSize = nParams//nActions
  stateFeatureVector = np.zeros(nParams, dtype=int)
  mappedIdx = int(mapValues(state, 0, nStates, 0, nParams))
  if mappedIdx>=0 and mappedIdx<nParams:
    stateFeatureVector[mappedIdx] = 1
  featureVector = np.zeros(nParams)
  featureVector[idx_action*stateFeatureVectorSize:(idx_action+1)*stateFeatureVectorSize] = stateFeatureVector
  return featureVector
  
def polynomial(state, action=None, **kwargs):
  nParams = getValueFromDict(kwargs, "nParams")
  stateNormFactor = getValueFromDict(kwargs, "stateNormFactor", 1.0)
  c = getValueFromDict(kwargs, "c")
  if action is None:
    nActions = 1
    idx_action = 0
  else:
    nActions = getValueFromDict(kwargs, "nActions")
    idx_action = action
  stateFeatureVectorSize = nParams//nActions
  stateVector = buildStateVector(state*stateNormFactor)
  stateFeatureVector = np.ones(stateFeatureVectorSize)
  for i in range(stateFeatureVectorSize):
    for j in range(len(stateVector)):
      stateFeatureVector[i]*=stateVector[j]**c[i,j]
  featureVector = np.zeros(nParams)
  featureVector[idx_action*stateFeatureVectorSize:(idx_action+1)*stateFeatureVectorSize] = stateFeatureVector
  return featureVector
  
def fourier(state, action=None, **kwargs):
  nParams = getValueFromDict(kwargs, "nParams") 
  stateNormFactor = getValueFromDict(kwargs, "stateNormFactor", 1.0)
  if action is None:
    nActions = 1
    idx_action = 0
  else:
    nActions = getValueFromDict(kwargs, "nActions")
    idx_action = action
  stateFeatureVectorSize = nParams//nActions
  stateFeatureVector = np.array([np.cos(i*np.pi*state*stateNormFactor) for i in range(stateFeatureVectorSize)], dtype=float)
  featureVector = np.zeros(nParams)
  featureVector[idx_action*stateFeatureVectorSize:(idx_action+1)*stateFeatureVectorSize] = stateFeatureVector
  return featureVector
  
def tileCoding(state, action=None, **kwargs):
  '''
  Tile coding with grid tiles
  ''' 
  minStates = np.array(getValueFromDict(kwargs, "minStates"))
  maxStates = np.array(getValueFromDict(kwargs, "maxStates"))
  nTilings = getValueFromDict(kwargs, "nTilings")
  tilingOffsets = np.array(getValueFromDict(kwargs, "tilingOffsets"))
  tilingSize = np.array(getValueFromDict(kwargs, "tilingSize"))
  dimStates = len(minStates)
  if action is None:
    nActions = 1
    idx_action = 0
  else:
    nActions = getValueFromDict(kwargs, "nActions")
    idx_action = action
  stateVector = buildStateVector(state)
  stateFeatureVector = []
  for idx_tiling in range(nTilings):
    tileVector = np.zeros(tilingSize[idx_tiling])
    mappedIdx = mapValues(stateVector+tilingOffsets[idx_tiling], minStates, maxStates, np.zeros(dimStates), tilingSize[idx_tiling])
    mappedIdx = np.array(mappedIdx, dtype=int)
    if min(mappedIdx>=np.zeros_like(mappedIdx))==True and min(mappedIdx<tilingSize[idx_tiling])==True:
      tileVector[tuple(mappedIdx)] = 1
    stateFeatureVector.extend(tileVector.flatten())
  stateFeatureVectorSize = len(stateFeatureVector)
  featureVector = np.zeros(nActions*stateFeatureVectorSize)
  featureVector[idx_action*stateFeatureVectorSize:(idx_action+1)*stateFeatureVectorSize] = stateFeatureVector
  return np.array(featureVector, dtype=int).flatten()
  
def radialBasisFunction(state, action=None, **kwargs):
  mu = getValueFromDict(kwargs, "mu")
  sigma = getValueFromDict(kwargs, "sigma")
  if action is None:
    nActions = 1
    idx_action = 0
  else:
    nActions = getValueFromDict(kwargs, "nActions")
    idx_action = action
  stateFeatureVector = np.exp(-((state-mu)**2)/(2*sigma**2))
  stateFeatureVectorSize = len(stateFeatureVector)
  featureVector = np.zeros(stateFeatureVectorSize*nActions)
  featureVector[idx_action*stateFeatureVectorSize:(idx_action+1)*stateFeatureVectorSize] = stateFeatureVector
  return featureVector
  
def FixedStateEncoding(state, action=None, **kwargs):
  stateEncodingMatrix = getValueFromDict(kwargs, 'stateEncodingMatrix')
  if action is None:
    return stateEncodingMatrix[state,:].T
  else:
    return stateEncodingMatrix[state,action,:].T
