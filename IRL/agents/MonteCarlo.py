'''
MonteCarlo.py : implementations of Monte Carlo based algorithms

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
from IRL.utils.Policies import StochasticPolicy, selectAction_greedy

class MonteCarloPredictionVanilla:
  
  def __init__(self, nStates, gamma, doUseAllVisits=False):
    self.name = "Monte Carlo Prediction Vanilla"
    self.nStates = nStates
    self.gamma = gamma
    self.doUseAllVisits = doUseAllVisits
    self.returns = {}
    self.valueTable = np.zeros([self.nStates], dtype=float)
    
  def evaluate(self, episode):
    T = len(episode)-1
    G = 0.0
    states = [memory["state"] for memory in episode]
    for t in range(T-1, -1, -1):
      state = episode[t]["state"]
      reward = episode[t+1]["reward"]
      G = self.gamma * G + reward
      if(self.doUseAllVisits):
        doUpdate = True
      else:
        doUpdate = (state not in states[0:t])
      if(doUpdate):
        if state not in self.returns.keys():
          self.returns[state] = []
        self.returns[state].append(G)
        self.valueTable[state] = np.mean(self.returns[state])
    
  def reset(self):
    self.valueTable = np.zeros([self.nStates], dtype=float)
    self.returns = {}

  def getValue(self, state):
    return self.valueTable[state]
    
  def getName(self):
    return self.name
    
class MonteCarloPrediction:
  '''
  Incremental implementation with fixed weight option
  '''
  
  def __init__(self, nStates, gamma, alpha=None, doUseAllVisits=False):
    self.name = "Monte Carlo Prediction"
    self.nStates = nStates
    self.gamma = gamma
    self.alpha = alpha
    self.doUseAllVisits = doUseAllVisits
    self.visitCounts = np.zeros([self.nStates], dtype=int)
    self.valueTable = np.zeros([self.nStates], dtype=float)
    
  def evaluate(self, episode):
    T = len(episode)-1
    G = 0.0
    states = [memory["state"] for memory in episode]
    for t in range(T-1, -1, -1):
      state = episode[t]["state"]
      reward = episode[t+1]["reward"]
      G = self.gamma * G + reward
      if(self.doUseAllVisits):
        doUpdate = True
      else:
        doUpdate = (state not in states[0:t])
      if(doUpdate):
        self.visitCounts[state] += 1
        if(self.alpha is None):
          alpha = 1.0/self.visitCounts[state]
        else:
          alpha = self.alpha
        self.valueTable[state] = self.valueTable[state] + alpha * (G - self.valueTable[state])

  def getValue(self, state):
    return self.valueTable[state]
    
  def getName(self):
    return self.name
    
  def reset(self):
    self.visitCounts = np.zeros([self.nStates], dtype=int)
    self.valueTable = np.zeros([self.nStates], dtype=float)    

class MCControlAgent:

  def __init__(self, nStates, nActions, gamma, policyUpdateMethod="greedy", epsilon=0.0, tieBreakingMethod="arbitrary"):
    self.name = "Generic Monte Carlo Control Agent"
    self.nStates = nStates
    self.nActions = nActions
    self.gamma = gamma
    self.actionValueTable = np.zeros([self.nStates, self.nActions], dtype=float)
    self.policy = StochasticPolicy(self.nStates, self.nActions, policyUpdateMethod=policyUpdateMethod,
      epsilon=epsilon, tieBreakingMethod=tieBreakingMethod)

  def selectAction(self, state, actionsAvailable=None):
    return self.policy.sampleAction(state, actionsAvailable)
    
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
    self.actionValueTable = np.zeros([self.nStates, self.nActions], dtype=np.float)
    self.policy.reset()    
    
class MonteCarloControl(MCControlAgent):
  
  def __init__(self, nStates, nActions, gamma, doUseAllVisits=False, policyUpdateMethod="esoft", 
    epsilon=0.01, tieBreakingMethod="arbitrary"):
    super().__init__(nStates, nActions, gamma, policyUpdateMethod=policyUpdateMethod, 
    epsilon=epsilon, tieBreakingMethod=tieBreakingMethod)
    self.name = "Monte Carlo Control"
    self.doUseAllVisits = doUseAllVisits
    self.returns = {}
    
  def update(self, episode):
    T = len(episode)-1
    G = 0.0
    state_action_pairs = [(xp["state"], xp["action"]) for xp in episode[:-1]]
    for t in range(T-1, -1, -1):
      state = episode[t]["state"]
      action = episode[t]["action"]
      reward = episode[t+1]["reward"]
      G = self.gamma * G + reward
      if(self.doUseAllVisits):
        doUpdate = True
      else:
        doUpdate = ((state, action) not in state_action_pairs[0:t])
      if(doUpdate):
        if (state, action) not in self.returns.keys():
          self.returns[(state, action)] = []
        self.returns[(state, action)].append(G)
        self.actionValueTable[state, action] = np.mean(self.returns[(state, action)])
        self.policy.update(state, self.actionValueTable[state,:])
        
  def reset(self):
    super().reset()
    self.returns = {}
    
class MonteCarloOffPolicyPrediction(MCControlAgent):
  
  def __init__(self, nStates, nActions, gamma, doUseWeightedIS=True):
    super().__init__(nStates, nActions, gamma)
    self.name = "Monte Carlo Off-policy Prediction"
    self.doUseWeightedIS = doUseWeightedIS
    self.C = np.zeros([self.nStates, self.nActions], dtype=np.float)   
    self.returns = {}

  def evaluate(self, episode, behaviour_policy):
    if(self.doUseWeightedIS):
      return self.evaluate_weighted(episode, behaviour_policy)
    else:
      return self.evaluate_ordinary(episode, behaviour_policy)
      
  def evaluate_weighted(self, episode, behaviour_policy):
    T = len(episode)-1
    G = 0.0
    W = 1.0
    state_action_pairs = [(memory["state"], memory["action"]) for memory in episode[:-1]]
    for t in range(T-1, -1, -1):
      if(W==0):
        break
      state = episode[t]["state"]
      action = episode[t]["action"]
      reward = episode[t+1]["reward"]
      G = self.gamma * G + reward
      self.C[state, action] += W
      self.actionValueTable[state, action] = self.actionValueTable[state, action] + (W/self.C[state, action])*(G-self.actionValueTable[state, action])
      W = W * (self.policy.getProbability(state, action)/behaviour_policy.getProbability(state, action))

  def evaluate_ordinary(self, episode, behaviour_policy):
    T = len(episode)-1
    G = 0.0
    W = 1.0
    states = [memory["state"] for memory in episode]
    for t in range(T-1, -1, -1):
      if(W==0):
        break
      state = episode[t]["state"]
      action = episode[t]["action"]
      reward = episode[t+1]["reward"]
      G = self.gamma * G + reward
      if (state,action) not in self.returns.keys():
        self.returns[(state,action)] = []
      self.returns[(state,action)].append(W*G)
      W = W * (self.policy.getProbability(state, action)/behaviour_policy.getProbability(state, action))
      self.actionValueTable[state, action] = np.mean(self.returns[(state,action)])
    
  def reset(self):
    super().reset()
    self.returns = {}
    self.C = np.zeros([self.nStates, self.nActions], dtype=np.float)
    
class MonteCarloOffPolicyControl(MCControlAgent):

  def __init__(self, nStates, nActions, gamma, policyUpdateMethod="greedy", epsilon=0.0, tieBreakingMethod="consistent"):
    super().__init__(nStates, nActions, gamma, policyUpdateMethod=policyUpdateMethod, 
      epsilon=epsilon, tieBreakingMethod=tieBreakingMethod)
    self.name = "Monte Carlo Off-Policy Control"
    self.C = np.zeros([self.nStates, self.nActions], dtype=np.float)
    for idx_state in range(self.nStates):
      self.policy.update(idx_state, self.actionValueTable[idx_state,:])

  def update(self, episode, behaviour_policy):
    T = len(episode)-1
    G = 0.0
    W = 1.0
    for t in range(T-1, -1, -1):
      state = episode[t]["state"]
      action = episode[t]["action"]
      reward = episode[t+1]["reward"]
      if("allowedActions" in episode[t].keys()):
        allowedActions = episode[t]["allowedActions"]
      else:
        allowedActions = np.array(range(self.nActions))
      G = self.gamma * G + reward
      self.C[state, action] += W
      self.actionValueTable[state, action] += (W/self.C[state, action])*(G-self.actionValueTable[state, action])
      self.policy.update(state, self.actionValueTable[state,:])
      if action!=self.getGreedyAction(state, allowedActions):
        break
      W = W * (1.0/behaviour_policy.getProbability(state, action))
      
  def reset(self):
    super().reset()
    self.C = np.zeros([self.nStates, self.nActions], dtype=np.float)
    