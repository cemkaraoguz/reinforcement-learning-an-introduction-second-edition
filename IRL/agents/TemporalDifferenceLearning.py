'''
TemporalDifferenceLearning.py : Implementation of various Temporal Difference Learning based algorithms

Cem Karaoguz, 2020
MIT License
'''

import sys
import numpy as np
from queue import PriorityQueue
from IRL.utils.Policies import ActionValuePolicy, StochasticPolicy, selectAction_greedy
from IRL.agents.Planning import DeterministicModel
from IRL.utils import Numeric

# ----------------------------
#      Classical Methods
# ----------------------------

class TDPrediction:
  
  def __init__(self, nStates, alpha, gamma, valueInit="zeros"):
    self.name = "TD Prediction"
    self.nStates = nStates
    self.alpha = alpha
    self.gamma = gamma
    self.valueInit = valueInit
    self.updates = np.zeros([self.nStates], dtype=float)
    if(self.valueInit=="zeros"):
      self.valueTable = np.zeros([self.nStates], dtype=float)
    elif(self.valueInit=="random"):
      self.valueTable = np.random.rand(self.nStates)
    else:
      sys.exit("ERROR: TDPrediction: valueInit not recognized!") 
      
  def evaluate(self, episode):
    T = len(episode)
    for t in range(0, T-1):
      state = episode[t]["state"]
      reward = episode[t+1]["reward"]
      next_state = episode[t+1]["state"]
      done = episode[t+1]["done"]
      td_error = reward + self.gamma * self.valueTable[next_state] - self.valueTable[state]
      self.valueTable[state] += self.alpha * td_error
    
  def getValue(self, state):
    return self.valueTable[state]
    
  def getName(self):
    return self.name
    
  def reset(self):
    if(self.valueInit=="zeros"):
      self.valueTable = np.zeros([self.nStates], dtype=float)
    elif(self.valueInit=="random"):
      self.valueTable = np.random.rand(self.nStates)
    else:
      sys.exit("ERROR: TDPrediction: valueInit not recognized!")

class TDControlAgent:

  def __init__(self, nStates, nActions, alpha, gamma, actionSelectionMethod="egreedy", epsilon=0.01, 
    tieBreakingMethod="arbitrary", valueInit="zeros"):    
    self.name = "Generic TDControlAgent"
    self.nStates = nStates
    self.nActions = nActions
    self.alpha = alpha
    self.gamma = gamma
    self.valueInit = valueInit
    if(self.valueInit=="zeros"):
      self.actionValueTable = np.zeros([self.nStates, self.nActions], dtype=float)
    elif(self.valueInit=="random"):
      self.actionValueTable = np.random.rand(self.nStates, self.nActions)
    else:
      sys.exit("ERROR: TDControlAgent: valueInit not recognized!")    
    self.policy = ActionValuePolicy(self.nStates, self.nActions, actionSelectionMethod=actionSelectionMethod, 
      epsilon=epsilon, tieBreakingMethod=tieBreakingMethod)
    for idx_state in range(self.nStates):
      self.policy.update(idx_state, self.actionValueTable[idx_state,:])
      
  def selectAction(self, state, actionsAvailable=None):
    return self.policy.selectAction(state, actionsAvailable)
    
  def getGreedyAction(self, state, actionsAvailable=None):
    if(actionsAvailable is None):
      actionValues = self.actionValueTable[state,:]
      actionList = np.array(range(self.nActions))
    else:
      actionValues = self.actionValueTable[state, actionsAvailable]
      actionList = np.array(actionsAvailable)
    actionIdx = selectAction_greedy(actionValues)
    return actionList[actionIdx]
  
  def getValue(self, state):
    return np.dot(self.policy.getProbability(state), self.actionValueTable[state,:])
    
  def getActionValue(self, state, action):
    return self.actionValueTable[state,action]
    
  def getName(self):
    return self.name
    
  def reset(self):
    if(self.valueInit=="zeros"):
      self.actionValueTable = np.zeros([self.nStates, self.nActions], dtype=float)
    elif(self.valueInit=="random"):
      self.actionValueTable = np.random.rand(self.nStates, self.nActions)
    else:
      sys.exit("ERROR: TDControlAgent: valueInit not recognized!")    
    self.policy.reset()
    for idx_state in range(self.nStates):
      self.policy.update(idx_state, self.actionValueTable[idx_state,:])    
    
class SARSA(TDControlAgent):
  
  def __init__(self, nStates, nActions, alpha, gamma, actionSelectionMethod="egreedy", 
    epsilon=0.01, tieBreakingMethod="arbitrary", valueInit="zeros"):
    super().__init__(nStates, nActions, alpha, gamma, actionSelectionMethod=actionSelectionMethod, 
      epsilon=epsilon, tieBreakingMethod=tieBreakingMethod, valueInit=valueInit)
    self.name = "SARSA"

  def update(self, episode):
    T = len(episode)
    for t in range(0, T-1):
      state = episode[t]["state"]
      action = episode[t]["action"]
      reward = episode[t+1]["reward"]
      next_state = episode[t+1]["state"]
      next_action = episode[t+1]["action"]
      td_error = reward + self.gamma * self.actionValueTable[next_state, next_action] - self.actionValueTable[state, action]
      self.actionValueTable[state, action] += self.alpha * td_error
      self.policy.update(state, self.actionValueTable[state,:])

class ExpectedSARSA(TDControlAgent):
  
  def __init__(self, nStates, nActions, alpha, gamma, actionSelectionMethod="esoft", 
    epsilon=0.01, tieBreakingMethod="arbitrary", valueInit="zeros"):
    super().__init__(nStates, nActions, alpha, gamma, valueInit=valueInit)
    self.name = "Expected SARSA"
    self.policy = StochasticPolicy(self.nStates, self.nActions, policyUpdateMethod="esoft", 
      epsilon=epsilon, tieBreakingMethod=tieBreakingMethod)
    
  def update(self, episode):
    T = len(episode)
    for t in range(0, T-1):
      state = episode[t]["state"]
      action = episode[t]["action"]
      reward = episode[t+1]["reward"]
      next_state = episode[t+1]["state"]
      if("allowedActions" in episode[t+1].keys()):
        allowedActions = episode[t+1]["allowedActions"]
        pdist = Numeric.normalize_sum(self.policy.getProbability(next_state)[allowedActions])
      else:
        allowedActions = np.array(range(self.nActions))
        pdist = self.policy.getProbability(next_state)
      expectedVal = np.dot(pdist, self.actionValueTable[next_state, allowedActions])
      td_error = reward + self.gamma * expectedVal - self.actionValueTable[state, action]
      self.actionValueTable[state, action] += self.alpha * td_error
      self.policy.update(state, self.actionValueTable[state,:])

  def selectAction(self, state, actionsAvailable=None):
    return self.policy.sampleAction(state, actionsAvailable)

class QLearning(TDControlAgent):
  
  def __init__(self, nStates, nActions, alpha, gamma, actionSelectionMethod="egreedy", epsilon=0.01, 
    tieBreakingMethod="arbitrary", valueInit="zeros"):
    super().__init__(nStates, nActions, alpha, gamma, actionSelectionMethod, epsilon=epsilon, 
      tieBreakingMethod=tieBreakingMethod, valueInit=valueInit)
    self.name = "Q-Learning"
  
  def update(self, episode):
    maxTDError = 0.0
    T = len(episode)
    for t in range(0, T-1):
      state = episode[t]["state"]
      action = episode[t]["action"]
      reward = episode[t+1]["reward"]
      next_state = episode[t+1]["state"]
      if("allowedActions" in episode[t+1].keys()):
        allowedActions = episode[t+1]["allowedActions"]
      else:
        allowedActions = np.array(range(self.nActions))
      td_error = reward + self.gamma * np.max(self.actionValueTable[next_state,allowedActions]) - self.actionValueTable[state, action]
      self.actionValueTable[state, action] += self.alpha * td_error
      self.policy.update(state, self.actionValueTable[state,:])
      maxTDError = max(maxTDError, abs(td_error))
    return maxTDError
    
class DoubleQLearning(TDControlAgent):
  
  def __init__(self, nStates, nActions, alpha, gamma, actionSelectionMethod="egreedy", epsilon=0.01, 
    tieBreakingMethod="arbitrary", valueInit="zeros"):
    self.name = "Double Q-Learning"
    self.nStates = nStates
    self.nActions = nActions
    self.alpha = alpha
    self.gamma = gamma
    self.tieBreakingMethod = tieBreakingMethod
    self.valueInit = valueInit
    if(self.tieBreakingMethod=="arbitrary"):
      self.argmax_function = Numeric.argmax
    elif(self.tieBreakingMethod=="consistent"):
      self.argmax_function = np.argmax
    else:
      sys.exit("ERROR: DoubleQLearning: tieBreakingMethod not recognized!")
    if(self.valueInit=="zeros"):
      self.actionValueTable_1 = np.zeros([self.nStates, self.nActions], dtype=float)
      self.actionValueTable_2 = np.zeros([self.nStates, self.nActions], dtype=float)
    elif(self.valueInit=="random"):
      self.actionValueTable_1 = np.random.rand(self.nStates, self.nActions)
      self.actionValueTable_2 = np.random.rand(self.nStates, self.nActions)
    else:
      sys.exit("ERROR: DoubleQLearning: valueInit not recognized!")    
    self.policy = ActionValuePolicy(self.nStates, self.nActions, 
      actionSelectionMethod=actionSelectionMethod, epsilon=epsilon, tieBreakingMethod=tieBreakingMethod)
    for idx_state in range(self.nStates):
      self.policy.update(idx_state, self.actionValueTable_1[idx_state,:]+self.actionValueTable_2[idx_state,:])              
  
  def update(self, episode):
    T = len(episode)
    for t in range(0, T-1):
      state = episode[t]["state"]
      action = episode[t]["action"]
      reward = episode[t+1]["reward"]
      next_state = episode[t+1]["state"]
      if("allowedActions" in episode[t+1].keys()):
        allowedActions = episode[t+1]["allowedActions"]
      else:
        allowedActions = np.array(range(self.nActions))
      if(np.random.rand()<0.5):
        next_action = self.argmax_function(self.actionValueTable_1[next_state, allowedActions])
        td_error = reward + self.gamma * self.actionValueTable_2[next_state, next_action] - self.actionValueTable_1[state, action]
        self.actionValueTable_1[state, action] += self.alpha * td_error
      else:
        next_action = self.argmax_function(self.actionValueTable_2[next_state,allowedActions])
        td_error = reward + self.gamma * self.actionValueTable_1[next_state,next_action] - self.actionValueTable_2[state, action]
        self.actionValueTable_2[state, action] += self.alpha * td_error
      self.policy.update(state, (self.actionValueTable_1[state,:]+self.actionValueTable_2[state,:]))
    
  def getValue(self, state):
    q_values = self.actionValueTable_1[state,:] + self.actionValueTable_2[state,:]
    return np.dot(self.policy.getProbability(state), q_values)
    
  def getActionValue(self, state, action):
    return self.actionValueTable_1[state,action] + self.actionValueTable_2[state,action]
    
  def getGreedyAction(self, state, actionsAvailable=None):
    actionValueTable = (self.actionValueTable_1 + self.actionValueTable_2)/2.0
    if(actionsAvailable is None):
      actionValues = actionValueTable[state,:]
      actionList = np.array(range(self.nActions))
    else:
      actionValues = actionValueTable[state, actionsAvailable]
      actionList = np.array(actionsAvailable)   
    actionIdx = selectAction_greedy(actionValues)
    return actionList[actionIdx]
    
  def reset(self):
    if(self.valueInit=="zeros"):
      self.actionValueTable_1 = np.zeros([self.nStates, self.nActions], dtype=float)
      self.actionValueTable_2 = np.zeros([self.nStates, self.nActions], dtype=float)
    elif(self.valueInit=="random"):
      self.actionValueTable_1 = np.random.rand(self.nStates, self.nActions)
      self.actionValueTable_2 = np.random.rand(self.nStates, self.nActions)
    else:
      sys.exit("ERROR: DoubleQLearning: valueInit not recognized!")    
    self.policy.reset()
    for idx_state in range(self.nStates):
      self.policy.update(idx_state, self.actionValueTable_1[idx_state,:]+self.actionValueTable_2[idx_state,:])              

# ----------------------------
#        n-step Methods
# ----------------------------

class nStepTDPredictionAgent:
  
  def __init__(self, nStates, alpha, gamma, n, valueInit="zeros"):
    self.name = "Generic n-step TD Prediction Agent"
    self.nStates = nStates
    self.alpha = alpha
    self.gamma = gamma
    self.n = n
    self.valueInit = valueInit
    if(self.valueInit=="zeros"):
      self.valueTable = np.zeros([self.nStates], dtype=float)
    elif(self.valueInit=="random"):
      self.valueTable = np.random.rand(self.nStates)
    else:
      sys.exit("ERROR: nStepTDPredictionAgent: valueInit not recognized!") 
    self.bufferExperience = []
  
  def evaluate(self, episode, behaviour_policy=None):
    self.updateBuffer(episode)
    t = len(self.bufferExperience)-2
    tau = t + 1 - self.n
    if self.bufferExperience[t+1]['done']:
      T = t+1
      self.sweepBuffer(max(0,tau), T, t, T, behaviour_policy)
      self.bufferExperience = []
    elif tau>=0:
      T = np.inf    
      self.sweepBuffer(tau, tau+1, t, T, behaviour_policy)
      
  def updateBuffer(self, episode):
    self.bufferExperience.extend(episode)
    while(len(self.bufferExperience)>(self.n+1)):
      self.bufferExperience.pop(0)
    
  def getValue(self, state):
    return self.valueTable[state]

  def getName(self):
    return self.name

  def reset(self):
    if(self.valueInit=="zeros"):
      self.valueTable = np.zeros([self.nStates], dtype=float)
    elif(self.valueInit=="random"):
      self.valueTable = np.random.rand(self.nStates)
    else:
      sys.exit("ERROR: nStepTDPredictionAgent: valueInit not recognized!") 
    self.bufferExperience = []
    
class nStepTDPrediction(nStepTDPredictionAgent):
  
  def __init__(self, nStates, alpha, gamma, n, valueInit="zeros"):
    super().__init__(nStates, alpha, gamma, n, valueInit=valueInit)
    self.name = "n-step TD Prediction"

  def sweepBuffer(self, tau_start, tau_stop, t, T, behaviour_policy=None):
    for tau in range(tau_start, tau_stop):
      state = self.bufferExperience[tau]['state']
      rewards = np.array([self.bufferExperience[i]['reward'] for i in range(tau+1, min(t+1, T)+1)])
      gammas = np.array([self.gamma**i for i in range(min(self.n-1, T-tau-1)+1)])
      G = np.dot(rewards,gammas)      
      if (tau+self.n)<T:
        G += self.gamma**self.n * self.getValue(self.bufferExperience[tau+self.n]['state'])
      td_error = G - self.valueTable[state]
      self.valueTable[state] += self.alpha * td_error

class nStepOffPolicyTDPrediction(nStepTDPredictionAgent):
  
  def __init__(self, nStates, nActions, alpha, gamma, n, valueInit="zeros", policyUpdateMethod="greedy", 
    epsilon=0.0, tieBreakingMethod="consistent"):
    super().__init__(nStates, alpha, gamma, n, valueInit=valueInit)
    self.name = "n-step Off-Policy TD Prediction"
    self.nActions = nActions
    self.policy = StochasticPolicy(self.nStates, self.nActions, policyUpdateMethod=policyUpdateMethod, 
      epsilon=epsilon, tieBreakingMethod=tieBreakingMethod)    

  def sweepBuffer(self, tau_start, tau_stop, t, T, behaviour_policy):
    for tau in range(tau_start, tau_stop):
      state = self.bufferExperience[tau]['state']
      l = min(T+1, t+1)
      G = self.valueTable[self.bufferExperience[l]['state']]
      W = 1.0
      for k in range(l-1, tau-1, -1):
        sweeping_state = self.bufferExperience[k]['state']
        sweeping_action = self.bufferExperience[k]['action']
        sweeping_reward = self.bufferExperience[k+1]['reward']
        p = self.policy.getProbability(sweeping_state, sweeping_action)
        b = behaviour_policy.getProbability(sweeping_state, sweeping_action)
        W = W * p/b
        G = self.gamma * G + sweeping_reward
      td_error = G - self.valueTable[state]
      self.valueTable[state] = self.valueTable[state] + self.alpha * td_error
      
  def reset(self):
    super().reset()
    self.policy.reset()
    
class nStepPerDecisionTDPrediction(nStepTDPredictionAgent):
  
  def __init__(self, nStates, nActions, alpha, gamma, n, valueInit="zeros", policyUpdateMethod="greedy", 
    epsilon=0.0, tieBreakingMethod="consistent"):
    super().__init__(nStates, alpha, gamma, n, valueInit=valueInit)
    self.name = "n-step Per-Decision TD Prediction"
    self.nActions = nActions
    self.policy = StochasticPolicy(self.nStates, self.nActions, policyUpdateMethod=policyUpdateMethod, 
      epsilon=epsilon, tieBreakingMethod=tieBreakingMethod)

  def sweepBuffer(self, tau_start, tau_stop, t, T, behaviour_policy):
    for tau in range(tau_start, tau_stop):
      state = self.bufferExperience[tau]['state']
      l = min(T+1, t+1)
      G = self.valueTable[self.bufferExperience[l]['state']]
      for k in range(l-1, tau-1, -1):
        sweeping_state = self.bufferExperience[k]['state']
        sweeping_action = self.bufferExperience[k]['action']
        sweeping_reward = self.bufferExperience[k+1]['reward']
        p = self.policy.getProbability(sweeping_state, sweeping_action)
        b = behaviour_policy.getProbability(sweeping_state, sweeping_action)
        W = p/b
        G = W * (sweeping_reward + self.gamma * G) + (1.0 - W)*self.valueTable[sweeping_state]
      td_error = G - self.valueTable[state]
      self.valueTable[state] = self.valueTable[state] + self.alpha * td_error

  def reset(self):
    super().reset()
    self.policy.reset()
    
class nStepTDControlAgent(TDControlAgent):

  def __init__(self, nStates, nActions, alpha, gamma, n, actionSelectionMethod="egreedy", epsilon=0.01, 
    tieBreakingMethod="arbitrary", valueInit="zeros"):
    super().__init__(nStates, nActions, alpha, gamma, actionSelectionMethod=actionSelectionMethod, epsilon=epsilon, 
      tieBreakingMethod=tieBreakingMethod,valueInit=valueInit)
    self.name = "Generic n-step TD Control Agent"
    self.n = n
    self.bufferExperience = []

  def update(self, episode, behaviour_policy=None):
    self.updateBuffer(episode)
    t = len(self.bufferExperience)-2
    tau = t + 1 - self.n
    if self.bufferExperience[t+1]['done']:
      T = t+1
      self.sweepBuffer(max(0,tau), T, t, T, behaviour_policy)
      self.bufferExperience = []
    elif tau>=0:
      T = np.inf    
      self.sweepBuffer(tau, tau+1, t, T, behaviour_policy)

  def updateBuffer(self, episode):
    self.bufferExperience.extend(episode)
    while(len(self.bufferExperience)>(self.n+1)):
      self.bufferExperience.pop(0)
    
class nStepSARSA(nStepTDControlAgent):
  
  def __init__(self, nStates, nActions, alpha, gamma, n, actionSelectionMethod="egreedy", epsilon=0.01, 
    tieBreakingMethod="arbitrary", valueInit="zeros"):
    super().__init__(nStates, nActions, alpha, gamma, n, actionSelectionMethod=actionSelectionMethod, epsilon=epsilon, 
      tieBreakingMethod=tieBreakingMethod, valueInit=valueInit)
    self.name = "n-step SARSA"
    
  def sweepBuffer(self, tau_start, tau_stop, t, T, behaviour_policy=None):
    for tau in range(tau_start, tau_stop):
      state = self.bufferExperience[tau]['state']
      action = self.bufferExperience[tau]['action']
      rewards = np.array([self.bufferExperience[i]['reward'] for i in range(tau+1, min(tau+self.n, t+1)+1)])
      gammas = np.array([self.gamma**i for i in range(min(self.n, t+1-tau))])
      if((tau+self.n)>t+1):
        G = np.sum(rewards * gammas)
      else:
        G = np.sum(rewards * gammas) + self.gamma**(self.n) * self.actionValueTable[self.bufferExperience[tau+self.n]['state'], self.bufferExperience[tau+self.n]['action']]      
      td_error = G - self.actionValueTable[state, action]
      self.actionValueTable[state, action] += self.alpha * td_error
      self.policy.update(state, self.actionValueTable[state,:])

class nStepOffPolicySARSA(nStepTDControlAgent):
  
  def __init__(self, nStates, nActions, alpha, gamma, n, policyUpdateMethod="esoft", epsilon=0.1, 
    tieBreakingMethod="arbitrary", valueInit="zeros"):
    super().__init__(nStates, nActions, alpha, gamma, n, valueInit=valueInit)
    self.name = "n-step off-policy SARSA"
    self.policy = StochasticPolicy(self.nStates, self.nActions, policyUpdateMethod=policyUpdateMethod, epsilon=epsilon, 
      tieBreakingMethod=tieBreakingMethod)
    
  def sweepBuffer(self, tau_start, tau_stop, t, T, behaviour_policy):
    for tau in range(tau_start, tau_stop):
      state = self.bufferExperience[tau]['state']
      action = self.bufferExperience[tau]['action']
      rewards = np.array([self.bufferExperience[i]['reward'] for i in range(tau+1, min(tau+self.n, t+1)+1)])
      gammas = np.array([self.gamma**i for i in range(min(self.n, t+1-tau))])
      l = min(tau+self.n, t+1)+1
      p = [self.policy.getProbability(self.bufferExperience[i]['state'], self.bufferExperience[i]['action']) for i in range(tau+1, l)]
      b = [behaviour_policy.getProbability(self.bufferExperience[i]['state'], self.bufferExperience[i]['action']) for i in range(tau+1, l)]
      W = np.prod(np.array(p)/np.array(b))
      G = np.sum(rewards * gammas)
      if(tau+self.n)<=t+1:
        G += self.gamma**(self.n) * self.actionValueTable[self.bufferExperience[tau+self.n]['state'], self.bufferExperience[tau+self.n]['action']]
      td_error = G - self.actionValueTable[state, action]
      self.actionValueTable[state, action] = self.actionValueTable[state, action] + self.alpha * W * td_error
      self.policy.update(state, self.actionValueTable[state,:])
  
  def selectAction(self, state, actionsAvailable=None):
    return self.policy.sampleAction(state, actionsAvailable)

class nStepTreeBackup(nStepTDControlAgent):
  
  def __init__(self, nStates, nActions, alpha, gamma, n, policyUpdateMethod="esoft", epsilon=0.1, 
    tieBreakingMethod="arbitrary", valueInit="zeros"):
    super().__init__(nStates, nActions, alpha, gamma, n, valueInit=valueInit)
    self.name = "n-step Tree Backup"
    self.policy = StochasticPolicy(self.nStates, self.nActions, policyUpdateMethod=policyUpdateMethod, epsilon=epsilon, 
      tieBreakingMethod=tieBreakingMethod)
  
  def sweepBuffer(self, tau_start, tau_stop, t, T, behaviour_policy=None):
    for tau in range(tau_start, tau_stop):
      state = self.bufferExperience[tau]['state']
      action = self.bufferExperience[tau]['action']
      if(t+1)>=T:
        G = self.bufferExperience[T]['reward']
      else:
        last_state = self.bufferExperience[t+1]['state']
        last_reward = self.bufferExperience[t+1]['reward']
        G = last_reward + self.gamma * np.dot(self.policy.getProbability(last_state), self.actionValueTable[last_state,:])
      for k in range(min(t, T-1), tau, -1):
        sweeping_state = self.bufferExperience[k]['state']
        sweeping_action = self.bufferExperience[k]['action']
        sweeping_reward = self.bufferExperience[k]['reward']
        probActions = np.array(self.policy.getProbability(sweeping_state))
        probAction = probActions[sweeping_action]
        probActions[sweeping_action] = 0.0
        G = sweeping_reward + self.gamma * np.dot(probActions, self.actionValueTable[sweeping_state,:]) + self.gamma * probAction * G
      td_error = G - self.actionValueTable[state, action]
      self.actionValueTable[state, action] = self.actionValueTable[state, action] + self.alpha * td_error
      self.policy.update(state, self.actionValueTable[state,:])

  def selectAction(self, state, actionsAvailable=None):
    return self.policy.sampleAction(state, actionsAvailable)

class nStepQSigma(nStepTDControlAgent):
  
  def __init__(self, nStates, nActions, alpha, gamma, n, sigma, policyUpdateMethod="esoft", epsilon=0.1, 
    tieBreakingMethod="arbitrary", valueInit="zeros"):
    super().__init__(nStates, nActions, alpha, gamma, n, valueInit=valueInit)
    self.name = "n-step Q-sigma"
    self.sigma = sigma
    self.policy = StochasticPolicy(self.nStates, self.nActions, policyUpdateMethod=policyUpdateMethod, epsilon=epsilon, 
      tieBreakingMethod=tieBreakingMethod) # TODO
  
  def sweepBuffer(self, tau_start, tau_stop, t, T, behaviour_policy):
    for tau in range(tau_start, tau_stop):
      state = self.bufferExperience[tau]['state']
      action = self.bufferExperience[tau]['action']
      if((t+1)<T):
        G = self.actionValueTable[self.bufferExperience[t+1]['state'], self.bufferExperience[t+1]['action']]
      for k in range(t+1, tau, -1):         
        sweeping_state = self.bufferExperience[k]['state']
        sweeping_action = self.bufferExperience[k]['action']
        sweeping_reward = self.bufferExperience[k]['reward']
        if(k==T):
          G = sweeping_reward
        else:
          sigma = self.sigma
          probActions = np.array(self.policy.getProbability(sweeping_state))
          p = probActions[sweeping_action]
          b = behaviour_policy.getProbability(sweeping_state, sweeping_action)
          W = p/b
          V = np.dot(probActions, self.actionValueTable[sweeping_state,:])
          G = sweeping_reward + self.gamma*(sigma*W + (1.0 - sigma)*p) * (G - self.actionValueTable[sweeping_state, sweeping_action]) + self.gamma*V
      td_error = G - self.actionValueTable[state, action]
      self.actionValueTable[state, action] = self.actionValueTable[state, action] + self.alpha * td_error
      self.policy.update(state, self.actionValueTable[state,:])
    
  def selectAction(self, state, actionsAvailable=None):
    return self.policy.sampleAction(state, actionsAvailable)

# ----------------------------
#    Planning based Methods
# ----------------------------
 
class DynaQ(QLearning):
  
  def __init__(self, nStates, nActions, alpha, gamma, nPlanningSteps, actionSelectionMethod="egreedy", 
    epsilon=0.01, valueInit="zeros", kappa=0.0):
    super().__init__(nStates, nActions, alpha, gamma, actionSelectionMethod, epsilon, valueInit=valueInit)
    self.name = "DynaQ"
    self.nPlanningSteps = nPlanningSteps
    self.kappa = kappa
    self.model = DeterministicModel(self.nStates, self.nActions, self.kappa)
    
  def update(self, episode):
    nUpdates = 0
    maxTDError = 0.0
    maxTDError_local = super().update(episode)
    self.model.update(episode)
    maxTDError = max(maxTDError, maxTDError_local)
    nUpdates += 1
    for planningStep in range(self.nPlanningSteps):
      simulatedXP = self.model.sampleExperience()
      maxTDError_local = super().update(simulatedXP)
      maxTDError = max(maxTDError, maxTDError_local)
      nUpdates += 1 
    return maxTDError, nUpdates
      
class PrioritizedSweeping(TDControlAgent):
    
  def __init__(self, nStates, nActions, alpha, gamma, nPlanningSteps, theta, actionSelectionMethod="egreedy", 
    epsilon=0.01, valueInit="zeros"):
    super().__init__(nStates, nActions, alpha, gamma, actionSelectionMethod, epsilon, valueInit=valueInit)
    self.name = "Prioritized Sweeping"
    self.nPlanningSteps = nPlanningSteps
    self.theta = theta
    self.model = DeterministicModel(self.nStates, self.nActions, doExtendActions=True)
    self.PQueue = PriorityQueue()
    
  def update(self, episode):
    nUpdates = 0
    maxTDError = 0
    self.model.update(episode)
    T = len(episode)
    for t in range(T-2, T-1):     
      state = episode[t]["state"]
      action = episode[t]["action"]
      reward = episode[t+1]["reward"]
      next_state = episode[t+1]["state"]
      td_error_abs = abs(reward + self.gamma * np.max(self.actionValueTable[next_state,:]) - self.actionValueTable[state, action])
      if td_error_abs>self.theta:
        self.PQueue.put((-td_error_abs, state, action))
      for i in range(self.nPlanningSteps):
        if self.PQueue.empty():
          break
        _, state_simu, action_simu = self.PQueue.get()
        reward_simu, next_state_simu = self.model[state_simu, action_simu]
        td_error_simu = reward_simu + self.gamma * np.max(self.actionValueTable[next_state_simu,:]) - self.actionValueTable[state_simu, action_simu]
        self.actionValueTable[state_simu, action_simu] = self.actionValueTable[state_simu, action_simu] + self.alpha * td_error_simu
        self.policy.update(state_simu, self.actionValueTable[state_simu, :])
        nUpdates += 1
        maxTDError = max(maxTDError, td_error_simu)
        ancestors = self.model.getAncestors(state_simu)
        for ancestor in ancestors:
          state_ancestor = int(ancestor[0])
          action_ancestor = int(ancestor[1])
          reward_ancestor = ancestor[2]
          td_error_ancestor = reward_ancestor + self.gamma * np.max(self.actionValueTable[state_simu,:]) - self.actionValueTable[state_ancestor, action_ancestor]
          if abs(td_error_ancestor)>self.theta:
            self.PQueue.put((abs(-td_error_ancestor), state_ancestor, action_ancestor))            
    return maxTDError, nUpdates
