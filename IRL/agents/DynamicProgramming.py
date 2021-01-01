'''
DynamicProgramming.py : implementations of dynamic programming based algorithms

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
from IRL.utils.Policies import ActionValuePolicy, selectAction_greedy
from IRL.utils.Numeric import normalize_greedy
from IRL.utils.Helpers import getValueFromDict

LARGE_NEGATIVE_VALUE = -10000000.0

class PolicyEvaluation():
    
  def __init__(self, nStates, nActions, gamma, thresh_convergence, expectedValueFunction, actionsAllowed=None):
    self.name = "Policy Evaluation"
    self.nStates = nStates
    self.nActions = nActions
    self.gamma = gamma
    self.thresh_convergence = thresh_convergence
    self.computeExpectedValue = expectedValueFunction
    if(actionsAllowed is None):
      self.actionsAllowed = np.ones([self.nStates, self.nActions], dtype=int)
    else:
      self.actionsAllowed = actionsAllowed
    self.valueTable = np.zeros([self.nStates], dtype=float)
  
  def evaluate(self, policy):
    deltaMax = 0.0
    isConverged = False
    for idx_state in range(self.nStates):
      newValue = 0.0
      for idx_action in range(self.nActions):
        prob_action = policy.getProbability(idx_state, idx_action)
        sum_expect_nextstates = self.computeExpectedValue(idx_state, idx_action, self.valueTable, self.gamma)
        newValue += prob_action * sum_expect_nextstates
      deltaMax = np.max([abs(self.valueTable[idx_state]-newValue), deltaMax])
      self.valueTable[idx_state] = newValue
    if deltaMax<=self.thresh_convergence:
      isConverged = True
    else:
      isConverged = False
    
    return deltaMax, isConverged
    
  def reset(self):
    self.valueTable = np.zeros([self.nStates], dtype=float)

class PolicyIteration():

  def __init__(self, nStates, nActions, gamma, thresh_convergence, expectedValueFunction, 
    actionsAllowed=None, iterationsMax=10000, doLogValueTables=False):
    self.name = "Policy Iteration"
    self.nStates = nStates
    self.nActions = nActions
    self.gamma = gamma
    self.thresh_convergence = thresh_convergence
    self.computeExpectedValue = expectedValueFunction
    self.iterationsMax = iterationsMax
    if(actionsAllowed is None):
      self.actionsAllowed = np.ones([self.nStates, self.nActions], dtype=int)
    else:
      self.actionsAllowed = actionsAllowed
    self.doLogValueTables = doLogValueTables
    self.valueTable = np.zeros([self.nStates], dtype=float)
    self.policy = ActionValuePolicy(self.nStates, self.nActions, actionSelectionMethod="greedy")
    self.valueTables = []
  
  def update(self):
    for i in range(self.iterationsMax):
      
      print("Policy evaluation iteration : ", i)
      
      deltaMax, isConverged = self.evaluate()
      
      if(isConverged):
        break

    if self.doLogValueTables:
      self.valueTables.append(np.array(self.valueTable)) # For visualization
   
    if(isConverged):
      isPolicyStable = self.improve()
    else:
      isPolicyStable = False
     
    return deltaMax, isConverged, isPolicyStable
    
  def evaluate(self):
    deltaMax = 0.0
    isConverged = False
    for idx_state in range(self.nStates):
      availableActions = np.nonzero(self.actionsAllowed[idx_state,:])[0]
      idx_action = self.policy.selectAction(idx_state, availableActions)
      newValue = self.computeExpectedValue(idx_state, idx_action, self.valueTable, self.gamma)
      deltaMax = max(abs(self.valueTable[idx_state]-newValue), deltaMax)
      self.valueTable[idx_state] = newValue
    if deltaMax<=self.thresh_convergence:
      isConverged = True
    else:
      isConverged = False
      
    return deltaMax, isConverged
    
  def improve(self):
    isPolicyStable = True
    for idx_state in range(self.nStates):
      availableActions = np.nonzero(self.actionsAllowed[idx_state,:])[0]
      if(len(availableActions)==0):
        continue
      oldAction = self.policy.selectAction(idx_state, availableActions)
      actionValues = np.zeros(self.nActions)
      for idx_action in range(self.nActions):
        if(self.actionsAllowed[idx_state, idx_action]):
          sum_expect_nextstates = self.computeExpectedValue(idx_state, idx_action, self.valueTable, self.gamma)
          actionValues[idx_action] = sum_expect_nextstates
        else:
          actionValues[idx_action] = LARGE_NEGATIVE_VALUE
      maxValuedAction = np.argmax(actionValues)
      actionProb = np.zeros(self.nActions)
      actionProb[maxValuedAction] = 1.0
      self.policy.update(idx_state, actionProb)
      
      #if(oldAction!=maxValuedAction): # Buggy version from the book
      if(actionValues[oldAction]<np.max(actionValues)):
        isPolicyStable = False
        
    return isPolicyStable
    
  def selectAction(self, state, actionsAvailable=None):
    return self.policy.selectAction(state, actionsAvailable)
    
  def getValue(self, state):
    return self.valueTable[state]

  def getGreedyAction(self, state, actionsAvailable=None):
    if(actionsAvailable is None):
      actionValues = self.policy.actionValueTable[state,:]
      actionList = np.array(range(self.nActions))
    else:
      actionValues = self.policy.actionValueTable[state, actionsAvailable]
      actionList = np.array(actionsAvailable)
    actionIdx = selectAction_greedy(actionValues)
    return actionList[actionIdx]

  def reset(self):
    self.valueTable = np.zeros([self.nStates], dtype=float)
    self.policy.reset()
    
class ValueIteration():

  def __init__(self, nStates, nActions, gamma, thresh_convergence, expectedValueFunction, 
    actionsAllowed=None, iterationsMax=10000, doLogValueTables=False):
    self.name = "Value Iteration"
    self.nStates = nStates
    self.nActions = nActions
    self.gamma = gamma
    self.thresh_convergence = thresh_convergence
    self.computeExpectedValue = expectedValueFunction
    self.iterationsMax = iterationsMax
    if(actionsAllowed is None):
      self.actionsAllowed = np.ones([self.nStates, self.nActions], dtype=int)
    else:
      self.actionsAllowed = actionsAllowed
    self.doLogValueTables = doLogValueTables
    self.valueTable = np.zeros([self.nStates], dtype=float)
    self.policy = ActionValuePolicy(self.nStates, self.nActions, actionSelectionMethod="greedy")
    self.valueTables = []
  
  def update(self):
    for i in range(self.iterationsMax):
      
      print("Policy evaluation iteration : ", i)
      
      deltaMax, isConverged = self.evaluate()
      
      if self.doLogValueTables:
        self.valueTables.append(np.array(self.valueTable)) # For visualization
      
      if(isConverged):
        break
    
    if(isConverged):
      isPolicyStable = self.improve()
    else:
      isPolicyStable = False
    
    return deltaMax, isConverged, isPolicyStable
    
  def evaluate(self):
    deltaMax = 0.0
    isConverged = False
    for idx_state in range(self.nStates):
      availableActions = np.nonzero(self.actionsAllowed[idx_state,:])[0]
      maxActionVal = LARGE_NEGATIVE_VALUE
      maxActionIdx = 0
      for idx_action in availableActions:
        actionVal = self.computeExpectedValue(idx_state, idx_action, self.valueTable, self.gamma)     
        if(actionVal>=maxActionVal):
          maxActionVal = actionVal
          maxActionIdx = idx_action
      deltaMax = np.max([abs(self.valueTable[idx_state]-maxActionVal), deltaMax])
      self.valueTable[idx_state] = maxActionVal
    if deltaMax<=self.thresh_convergence:
      isConverged = True
    else:
      isConverged = False
    return deltaMax, isConverged
    
  def improve(self):
    isPolicyStable = True
    for idx_state in range(self.nStates):
      availableActions = np.nonzero(self.actionsAllowed[idx_state,:])[0]
      if(len(availableActions)==0):
        continue
      oldAction = self.policy.selectAction(idx_state, availableActions)
      maxActionVal = LARGE_NEGATIVE_VALUE
      maxActionIdx = 0
      for idx_action in availableActions:
        actionVal = self.computeExpectedValue(idx_state, idx_action, self.valueTable, self.gamma)
        if(actionVal>=maxActionVal):
          maxActionVal = actionVal
          maxActionIdx = idx_action
      actionProb = np.zeros(self.nActions)
      actionProb[maxActionIdx] = 1.0
      self.policy.update(idx_state, actionProb)
    return isPolicyStable

  def selectAction(self, state, actionsAvailable=None):
    return self.policy.selectAction(state, actionsAvailable)

  def getValue(self, state):
    return self.valueTable[state]
    
  def getGreedyAction(self, state, actionsAvailable=None):
    if(actionsAvailable is None):
      actionValues = self.policy.actionValueTable[state,:]
      actionList = np.array(range(self.nActions))
    else:
      actionValues = self.policy.actionValueTable[state, actionsAvailable]
      actionList = np.array(actionsAvailable)
    actionIdx = selectAction_greedy(actionValues)
    return actionList[actionIdx]

  def reset(self):
    self.valueTable = np.zeros([self.nStates], dtype=float)
    self.policy.reset()

class ExpectedUpdateAgent:

  def __init__(self, nStates, nActions, gamma, expectedValueFunction, 
    actionSelectionMethod="egreedy", epsilon=0.1, tieBreakingMethod="arbitrary"):
    self.nStates = nStates
    self.nActions = nActions
    self.gamma = gamma
    self.computeExpectedUpdate = expectedValueFunction
    self.actionValueTable = np.zeros([self.nStates, self.nActions], dtype=float)
    self.policy = ActionValuePolicy(self.nStates, self.nActions, actionSelectionMethod=actionSelectionMethod, 
      epsilon=epsilon, tieBreakingMethod=tieBreakingMethod)
    self.policy.normalization_function = normalize_greedy
      
  def update(self, state, action):
    self.actionValueTable[state, action] = self.computeExpectedUpdate(state, action, self.actionValueTable)
    self.policy.update(state, self.actionValueTable[state,:])

  def selectAction(self, state):
    return self.policy.selectAction(state)
    
  def reset(self):
    self.actionValueTable = np.zeros([self.nStates, self.nActions], dtype=float)
    self.policy.reset()
    
class SemiGradientPolicyEvaluation():
    
  def __init__(self, nStates, nActions, nParams, gamma, alpha, thresh_convergence, 
    expectedValueFunction, approximationFunctionArgs):
    self.name = "Semi-Gradient Policy Evaluation"
    self.nStates = nStates
    self.nActions = nActions
    self.nParams = nParams
    self.gamma = gamma
    self.alpha = alpha
    self.thresh_convergence = thresh_convergence
    self.computeExpectedValue = expectedValueFunction
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.w = np.zeros([self.nParams], dtype=float)
    
  def evaluate(self, policy):
    deltaMax = 0.0
    isConverged = False
    delta = 0.0
    for idx_state in range(self.nStates):
      newValue = 0.0
      for idx_action in range(self.nActions):
        prob_action = policy.getProbability(idx_state, idx_action)
        sum_expect_nextstates = self.computeExpectedValue(idx_state, idx_action, self.w, self.af_kwargs, self.gamma)
        newValue += prob_action * sum_expect_nextstates
      deltaMax = np.max([abs(self.getValue(idx_state)-newValue), deltaMax])
      delta += ( newValue - self.getValue(idx_state) ) * self.afd(self.w, idx_state, **self.af_kwargs) 
    self.w += (self.alpha/self.nStates) * delta  
    if deltaMax<=self.thresh_convergence:
      isConverged = True
    else:
      isConverged = False
    return deltaMax, isConverged
    
  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
  
  def getName(self):
    return self.name

  def reset(self):
    self.w = np.zeros([self.nParams], dtype=float)

class ExpectedTDC():
    
  def __init__(self, nStates, nActions, nParams, gamma, alpha, beta, reward, approximationFunctionArgs):
    self.name = "Expected TDC"
    self.nStates = nStates
    self.nActions = nActions
    self.nParams = nParams
    self.gamma = gamma
    self.alpha = alpha
    self.beta = beta
    self.reward = reward
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.ftf = getValueFromDict(self.af_kwargs, "ftf")
    self.w = np.zeros([self.nParams], dtype=float)
    self.v = np.zeros([self.nParams], dtype=float)
    
  def evaluate(self, target_policy, behaviour_policy):
    delta_w = []
    delta_v = []
    for state in range(self.nStates):
      for action in range(self.nActions):
        for next_state in range(self.nStates):
          isr = target_policy.getProbability(state,action)/behaviour_policy.getProbability(state,action)
          x_t = self.ftf(state, **self.af_kwargs)
          x_tt = self.ftf(next_state, **self.af_kwargs) 
          td_error = self.reward + self.gamma * self.getValue(next_state) - self.getValue(state)
          delta_w.append( self.alpha * isr * (td_error * x_t - self.gamma * x_tt * np.dot(x_t.T, self.v)) )
          delta_v.append( self.beta * isr * (td_error - np.dot(self.v.T, x_t))*x_t )
    self.w += np.mean(delta_w, axis=0)
    self.v += np.mean(delta_v, axis=0)
  
  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)

  def getName(self):
    return self.name

  def reset(self):
    self.w = np.zeros([self.nParams], dtype=float)
    self.v = np.zeros([self.nParams], dtype=float)
    
class EmphaticTDPolicyEvaluation():
    
  def __init__(self, nStates, nActions, nParams, gamma, alpha, 
    thresh_convergence, expectedValueFunction, approximationFunctionArgs):
    self.name = "Emphatic TD"
    self.nStates = nStates
    self.nActions = nActions
    self.nParams = nParams
    self.gamma = gamma
    self.alpha = alpha
    self.thresh_convergence = thresh_convergence
    self.computeExpectedValue = expectedValueFunction
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.M = 0.0
    self.I = 1.0
    self.w = np.zeros([self.nParams], dtype=float)
    
  def evaluate(self, policy, importance=None):
    if importance is None:
      I = self.I
    else:
      I = importance
    deltaMax = 0.0
    isConverged = False
    delta = 0.0
    for idx_state in range(self.nStates):
      newValue = 0.0
      for idx_action in range(self.nActions):
        sum_expect_nextstates = self.computeExpectedValue(idx_state, idx_action, self.w, self.af_kwargs, self.gamma)
        newValue += policy.getProbability(idx_state, idx_action) * sum_expect_nextstates
      deltaMax = np.max([abs(self.getValue(idx_state)-newValue), deltaMax])
      td_error = newValue - self.getValue(idx_state) 
      delta += self.M * td_error * self.afd(self.w, idx_state, **self.af_kwargs)
    self.w += (self.alpha/self.nStates) * delta
    self.M = self.gamma * self.M + I
    if deltaMax<=self.thresh_convergence:
      isConverged = True
    else:
      isConverged = False
      
    return deltaMax, isConverged
    
  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
  
  def getName(self):
    return self.name
    
  def reset(self):
    self.M = 0.0
    self.I = 1.0
    self.w = np.zeros([self.nParams], dtype=float)
   