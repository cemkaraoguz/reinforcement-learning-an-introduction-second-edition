'''
Policies.py : implementations of various policies and action selection methods

Cem Karaoguz, 2020
MIT License
'''

import sys
import numpy as np

from IRL.utils import Numeric
from IRL.utils.Helpers import getValueFromDict

def selectAction_egreedy(actionValues, **kwargs):
  argmax_function = getValueFromDict(kwargs, "argmaxfun", np.argmax) 
  epsilon = kwargs["epsilon"]
  if np.random.rand()<epsilon:
    action = np.random.randint(0, len(actionValues))
  else:
    action = argmax_function(actionValues)
  return action

def selectAction_greedy(actionValues, **kwargs):
  argmax_function = getValueFromDict(kwargs, "argmaxfun", np.argmax)
  action = argmax_function(actionValues)
  return action

def selectAction_esoft(actionValues, **kwargs):
  argmax_function = getValueFromDict(kwargs, "argmaxfun", np.argmax)
  epsilon = kwargs["epsilon"]
  p = np.zeros_like(actionValues) + epsilon/(len(actionValues) - 1)
  p[argmax_function(actionValues)] = 1.0 - epsilon
  return np.random.choice(len(p), p=p)
  
def selectAction_esoft_(actionValues, **kwargs):
  # TODO: consider implementing this
  epsilon = kwargs["epsilon"]
  argmax_function = getValueFromDict(kwargs, "argmaxfun", np.argmax)
  q_max = np.max(actionValues) 
  n_greedy_actions = 0
  greedy_actions = []
  for i in range(len(actionValues)): 
    if actionValues[i] == q_max: 
      n_greedy_actions += 1
      greedy_actions.append(i)
  non_greedy_action_probability = epsilon / len(actionValues)
  greedy_action_probability = ((1.0 - epsilon) / n_greedy_actions) + non_greedy_action_probability 
  p=np.zeros(len(actionValues))+non_greedy_action_probability
  p[greedy_actions] = greedy_action_probability
  return np.random.choice(len(p), p=p)
  
def selectAction_softmax(actionValues, **kwargs):
  p = Numeric.normalize_softmax(actionValues)
  return np.random.choice(len(p), p=p)
  
def selectAction_UCB(actionValues, **kwargs):
  argmax_function = getValueFromDict(kwargs, "argmaxfun", np.argmax) 
  c = kwargs["c"]
  t = kwargs["t"]
  N = kwargs["N"]
  if np.min(N)==0:
    return np.argmin(N)
  else:
    return argmax_function(actionValues + c*np.sqrt(np.log(t)/N))
  
class DeterministicPolicy():
  '''
  Simple deterministic policy that maps states directly to actions:
    a = pi(s)
  '''
  
  def __init__(self, nStates, nActions, actionSelectionMethod="greedy", epsilon=0.0):
    self.nStates = nStates
    self.nActions = nActions
    self.actionSelectionMethod = actionSelectionMethod
    self.epsilon = epsilon
    self.stateActionTable = np.zeros([self.nStates], dtype=int)
    
  def selectAction(self, state, **kwargs):
    if self.actionSelectionMethod=="egreedy":
      if np.random.rand()<self.epsilon:
        return np.random.randint(0, self.nActions)
      else:
        return self.stateActionTable[state]
    elif self.actionSelectionMethod=="greedy":
      return self.stateActionTable[state]
    elif self.actionSelectionMethod=="UCB":
      return selectAction_UCB(kwargs["actionValues"][state], c=kwargs["c"], t=kwargs["t"], N=kwargs["N"][state])
    else:
      sys.exit("ERROR: DeterministicPolicy: actionSelectionMethod not recognized!")   

  def update(self, state, action):
    self.stateActionTable[state] = (int)(action)
   
  def reset(self):
    self.stateActionTable = np.zeros([self.nStates], dtype=int)    

class ActionValuePolicy():
  '''
  Deterministic policy based on action values:
    pi(s,a) ~ Q(s,a)
  '''
  
  def __init__(self, nStates, nActions, actionSelectionMethod="greedy", epsilon=0.0, tieBreakingMethod="consistent"):
    self.nStates = nStates
    self.nActions = nActions
    self.actionSelectionMethod = actionSelectionMethod
    self.epsilon = epsilon
    self.tieBreakingMethod = tieBreakingMethod
    if(self.tieBreakingMethod=="arbitrary"):
      self.argmax_function = Numeric.argmax
    elif(self.tieBreakingMethod=="consistent"):
      self.argmax_function = np.argmax
    else:
      sys.exit("ERROR: ActionValuePolicy: tieBreakingMethod not recognized!")
    if(self.actionSelectionMethod=="egreedy"):
      self.actionSelection_function = selectAction_egreedy
    elif(self.actionSelectionMethod=="softmax"):
      self.actionSelection_function = selectAction_softmax
    elif(self.actionSelectionMethod=="greedy"):
      self.actionSelection_function = selectAction_greedy
    elif(self.actionSelectionMethod=="esoft"):
      self.actionSelection_function = selectAction_esoft
    else:
      sys.exit("ERROR: ActionValuePolicy: actionSelectionMethod not recognized!")     
    self.actionValueTable = np.zeros([nStates, nActions], dtype=float)
    self.normalization_function = Numeric.normalize_sum
  
  def selectAction(self, state, actionsAvailable=None):   
    if(actionsAvailable is None):
      actionValues = self.actionValueTable[state,:]
      actionList = np.array(range(self.nActions))
    else:
      actionValues = self.actionValueTable[state, actionsAvailable]
      actionList = np.array(actionsAvailable)
    actionIdx = self.actionSelection_function(actionValues, argmaxfun=self.argmax_function, epsilon=self.epsilon)
    return actionList[actionIdx]

  def update(self, state, actionValues):
    self.actionValueTable[state,:] = actionValues
  
  def getProbability(self, state, action=None):
    p = self.normalization_function(self.actionValueTable[state,:], argmaxfun=self.argmax_function)
    if(action is None):
      return p
    else:
      return p[action]
  
  def reset(self):
    self.actionValueTable = np.zeros([nStates, nActions], dtype=float)

class StochasticPolicy():
  '''
  Stochastic policy that can be tied to learned action values:
    pi(a|s) ~ Q(s,a)
  '''
  
  def __init__(self, nStates, nActions, policyUpdateMethod="greedy", epsilon=0.0, tieBreakingMethod="consistent", initialDist=None):
    self.nStates = nStates
    self.nActions = nActions
    self.policyUpdateMethod = policyUpdateMethod
    self.epsilon = epsilon  
    self.tieBreakingMethod = tieBreakingMethod
    self.initialDist = initialDist
    if(self.tieBreakingMethod=="arbitrary"):
      self.argmax_function = Numeric.argmax
    elif(self.tieBreakingMethod=="consistent"):
      self.argmax_function = np.argmax
    else:
      sys.exit("ERROR: StochasticPolicy: tieBreakingMethod not recognized!")
    if(self.policyUpdateMethod=="normsum"):
      self.normalization_function = Numeric.normalize_sum
    elif(self.policyUpdateMethod=="softmax"):
      self.normalization_function = Numeric.normalize_softmax
    elif(self.policyUpdateMethod=="greedy"):
      self.normalization_function = Numeric.normalize_greedy
    elif(self.policyUpdateMethod=="esoft"):
      self.normalization_function = Numeric.normalize_esoft
    else:
      sys.exit("ERROR: StochasticPolicy: policyUpdateMethod not recognized!")
    if self.initialDist is None:
      self.actionProbabilityTable = np.ones([nStates, nActions], dtype=float) * 1.0/nActions
    else:
      self.actionProbabilityTable = self.initialDist
      for s in range(self.nStates):
        self.actionProbabilityTable[s,:] = Numeric.normalize_sum(self.actionProbabilityTable[s,:])
      
  def sampleAction(self, state, actionsAvailable=None):
    if(actionsAvailable is None):
      return np.random.choice(self.nActions, p=self.actionProbabilityTable[state,:])
    else:
      p = Numeric.normalize_sum(self.actionProbabilityTable[state, actionsAvailable])
      actionList = np.array(actionsAvailable)
      return np.random.choice(actionList, p=p)
    
  def update(self, state, actionValues):
    self.actionProbabilityTable[state,:] = self.normalization_function(actionValues, argmaxfun=self.argmax_function, epsilon=self.epsilon)   
  
  def getProbability(self, state, action=None):
    if(action is None):
      return self.actionProbabilityTable[state, :]
    else:
      return self.actionProbabilityTable[state, action]
      
  def reset(self):
    if self.initialDist is None:
      self.actionProbabilityTable = np.ones([nStates, nActions], dtype=float) * 1.0/nActions
    else:
      self.actionProbabilityTable = self.initialDist
      for s in range(self.nStates):
        self.actionProbabilityTable[s,:] = Numeric.normalize_sum(self.actionProbabilityTable[s,:])
        
class FunctionApproximationPolicy():
  '''
  Deterministic policy based on action values estimated via function approximation:
    pi(s,a) ~ Q(s,a|w)
  '''
  
  def __init__(self, nParams, nActions, approximationFunctionArgs, **kwargs):
    self.nParams = nParams
    self.nActions = nActions
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.actionSelectionMethod = getValueFromDict(kwargs, "actionSelectionMethod", "greedy")  
    self.epsilon = getValueFromDict(kwargs, "epsilon", 0.0)
    self.tieBreakingMethod = getValueFromDict(kwargs, "tieBreakingMethod", "consistent")
    self.w = np.zeros([self.nParams], dtype=float)
    
    if(self.tieBreakingMethod=="arbitrary"):
      self.argmax_function = Numeric.argmax
    elif(self.tieBreakingMethod=="consistent"):
      self.argmax_function = np.argmax
    else:
      sys.exit("ERROR: FunctionApproximationPolicy: tieBreakingMethod not recognized!")

    if(self.actionSelectionMethod=="egreedy"):
      self.actionSelection_function = selectAction_egreedy
    elif(self.actionSelectionMethod=="softmax"):
      self.actionSelection_function = selectAction_softmax
    elif(self.actionSelectionMethod=="greedy"):
      self.actionSelection_function = selectAction_greedy
    elif(self.actionSelectionMethod=="esoft"):
      self.actionSelection_function = selectAction_esoft
    else:
      sys.exit("ERROR: FunctionApproximationPolicy: actionSelectionMethod not recognized!")
      
  def selectAction(self, state):
    actionValues = np.array([self.af(self.w, state, action, **self.af_kwargs) for action in range(self.nActions)])
    actionIdx = self.actionSelection_function(actionValues, argmaxfun=self.argmax_function, epsilon=self.epsilon)
    return actionIdx
    
  def update(self, w):
    self.w[:] = w
    
  def getProbability(self, state, action=None):
    actionValues = np.array([self.af(self.w, state, a, **self.af_kwargs) for a in range(self.nActions)])
    pdist = Numeric.normalize_sum(actionValues)
    if(action is None):
      return pdist
    else:
      return pdist[action]
      
  def reset(self):
    self.w = np.zeros([self.nParams], dtype=float)
      
class ParametrizedPolicy:
  '''
  Parametrized stochastic policy that can be used with policy gradients:
    pi(a|s,theta)
  '''
  
  def __init__(self, nParams, nActions, approximationFunctionArgs, weightInit="random"):
    self.nParams = nParams
    self.nActions = nActions
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.weightInit = weightInit
    if self.weightInit=="zeros":
      self.theta = np.zeros([self.nParams], dtype=float)
    elif self.weightInit=="random":
      self.theta = np.random.randn(self.nParams)
    else:
      sys.exit("ERROR: ParametrizedPolicy: weightInit not recognized!")

  def sampleAction(self, state):
    return np.random.choice(self.nActions, p=self.getProbability(state))

  def grad(self, state, action):
    return self.afd(self.theta, state, action, **self.af_kwargs)
    
  def getProbability(self, state, action=None):
    return self.af(self.theta, state, action, **self.af_kwargs)
  
  def reset(self):
    if self.weightInit=="zeros":
      self.theta = np.zeros([self.nParams], dtype=float)
    elif self.weightInit=="random":
      self.theta = np.random.randn(self.nParams)
    else:
      sys.exit("ERROR: ParametrizedPolicy: weightInit not recognized!")
