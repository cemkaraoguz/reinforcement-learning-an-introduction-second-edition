'''
TemporalDifferenceApproximation.py : Implementation of function approximation based 
Temporal Difference Learning algorithms

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
from IRL.utils.Helpers import getValueFromDict
from IRL.utils.Policies import FunctionApproximationPolicy, selectAction_greedy

class SemiGradientTDPrediction:

  def __init__(self, nParams, alpha, gamma, approximationFunctionArgs):
    self.name = "Semi-gradient TD Prediction"
    self.nParams = nParams
    self.alpha = alpha
    self.gamma = gamma
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.w = np.zeros([self.nParams], dtype=float)
    
  def update(self, episode):  
    t = len(episode)-2
    state = episode[t]["state"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    done = episode[t+1]["done"]
    U = reward
    if not done:
      U += self.gamma * self.getValue(next_state)
    self.w += self.alpha*(U - self.getValue(state)) * self.afd(self.w, state, **self.af_kwargs)
    
  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
    
  def getName(self):
    return self.name
    
  def reset(self):
    self.w = np.zeros([self.nParams], dtype=float)
  
class SemiGradientOffPolicyTDPrediction:

  def __init__(self, nParams, alpha, gamma, targetPolicy, approximationFunctionArgs):
    self.name = "Semi-gradient Off-policy TD Prediction"
    self.nParams = nParams
    self.alpha = alpha
    self.gamma = gamma
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.policy = targetPolicy 
    self.w = np.zeros([self.nParams], dtype=float)
    
  def update(self, episode, behaviour_policy):  
    t = len(episode)-2
    state = episode[t]["state"]
    action = episode[t]["action"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    done = episode[t+1]["done"]
    isr = self.policy.getProbability(state,action)/behaviour_policy.getProbability(state,action)
    U = reward
    if not done:
      U += self.gamma * self.getValue(next_state)
    self.w += self.alpha*isr*(U - self.getValue(state)) * self.afd(self.w, state, **self.af_kwargs)
    
  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
    
  def getName(self):
    return self.name

  def reset(self):
    self.w = np.zeros([self.nParams], dtype=float)
    self.policy.reset()
    
class nStepSemiGradientTDPrediction:

  def __init__(self, nParams, alpha, gamma, n, approximationFunctionArgs):
    self.name = "Semi-gradient n-step TD Prediction"
    self.nParams = nParams
    self.alpha = alpha
    self.gamma = gamma
    self.n = n
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.w = np.zeros([self.nParams], dtype=float)
    self.bufferExperience = []
    
  def update(self, episode):    
    self.updateBuffer(episode)
    t = len(self.bufferExperience)-2
    tau = t + 1 - self.n
    if self.bufferExperience[t+1]['done']:
      T = t+1
      self.sweepBuffer(max(0,tau), T, t, T)
      self.bufferExperience = []
    elif tau>=0:
      T = np.inf    
      self.sweepBuffer(tau, tau+1, t, T)

  def sweepBuffer(self, tau_start, tau_stop, t, T):
    for tau in range(tau_start, tau_stop):
      state = self.bufferExperience[tau]['state']
      rewards = np.array([self.bufferExperience[i]['reward'] for i in range(tau+1, min(t+1, T)+1)])
      gammas = np.array([self.gamma**i for i in range(min(self.n-1, T-tau-1)+1)])
      G = np.sum(rewards * gammas)
      if (tau+self.n)<T:
        G += self.gamma**self.n * self.getValue(self.bufferExperience[tau+self.n]['state'])
      self.w += self.alpha*(G - self.getValue(state)) * self.afd(self.w, state, **self.af_kwargs)     

  def updateBuffer(self, episode):
    self.bufferExperience.extend(episode)
    while(len(self.bufferExperience)>(self.n+1)):
      self.bufferExperience.pop(0)
      
  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
    
  def getName(self):
    return self.name

  def reset(self):
    self.w = np.zeros([self.nParams], dtype=float)
    self.bufferExperience = []

class GradientTDPrediction:

  def __init__(self, nParams, alpha, beta, gamma, targetPolicy, approximationFunctionArgs):
    self.name = "Gradient TD Prediction"
    self.nParams = nParams
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.ftf = getValueFromDict(self.af_kwargs, "ftf")
    self.policy = targetPolicy
    self.w = np.zeros([self.nParams], dtype=float)
    self.v = np.zeros([self.nParams], dtype=float)
    
  def update(self, episode, behaviour_policy):  
    t = len(episode)-2
    state = episode[t]["state"]
    action = episode[t]["action"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    isr = self.policy.getProbability(state,action)/behaviour_policy.getProbability(state,action)
    x_t = self.ftf(state, **self.af_kwargs)
    x_tt = self.ftf(next_state, **self.af_kwargs) 
    td_error = reward + self.gamma * self.getValue(next_state) - self.getValue(state)
    self.w += self.alpha * isr * (td_error * x_t - self.gamma * x_tt * np.dot(x_t.T, self.v))
    self.v += self.beta * isr * (td_error - np.dot(self.v.T, x_t))*x_t
    
  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
    
  def getName(self):
    return self.name
    
  def reset(self):
    self.w = np.zeros([self.nParams], dtype=float)
    self.v = np.zeros([self.nParams], dtype=float)
    self.policy.reset()
    
class SemiGradientQLearningTDPrediction():
  
  def __init__(self, nParams, nActions, alpha, gamma, targetPolicy, approximationFunctionArgs):
    self.name = "Semi-gradient Q-Learning TD Prediction"
    self.nParams = nParams
    self.nActions = nActions
    self.alpha = alpha
    self.gamma = gamma
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.policy = targetPolicy
    self.w = np.zeros([self.nParams], dtype=float)

  def update(self, episode, behaviour_policy):  
    t = len(episode)-2
    state = episode[t]["state"]
    action = episode[t]["action"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    done = episode[t+1]["done"]
    isr = self.policy.getProbability(state,action)/behaviour_policy.getProbability(state,action)
    q_now = self.getValue(state, action)
    q_next = np.max(self.getValue(next_state))
    td_error = reward + self.gamma*q_next - q_now
    self.w += self.alpha * isr * td_error * self.afd(self.w, state, action, **self.af_kwargs)

  def getValue(self, state, action=None):
    if action is None:
      return np.array([self.af(self.w, state, a, **self.af_kwargs) for a in range(self.nActions)])
    else:
      return self.af(self.w, state, action, **self.af_kwargs)

  def getName(self):
    return self.name
    
  def reset(self):
    self.w = np.zeros([self.nParams], dtype=float)
    self.policy.reset()
  
class LeastSquaresTD:

  def __init__(self, nParams, gamma, epsilon, approximationFunctionArgs):
    self.name = "LSTD"
    self.nParams = nParams
    self.gamma = gamma
    self.epsilon = epsilon
    self.af_kwargs = approximationFunctionArgs
    self.ftf = getValueFromDict(self.af_kwargs, "ftf")
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.invA = np.eye(self.nParams)*self.epsilon
    self.b = np.zeros([self.nParams,1])
    self.w = np.zeros([self.nParams], dtype=float)
    
  def update(self, episode):
    t = len(episode)-2
    state = episode[t]["state"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    x = self.ftf(state, **self.af_kwargs).reshape([-1,1])
    x_next = self.ftf(next_state, **self.af_kwargs).reshape([-1,1])
    v = np.dot(self.invA.T, (x - self.gamma*x_next))
    self.invA -= np.dot( np.dot(self.invA, x), v.T)/(1 + np.dot(v.T, x))
    self.b += reward * x
    self.w = np.dot(self.invA, self.b)
    
  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
    
  def getName(self):
    return self.name
    
  def reset(self):
    self.invA = np.eye(self.nParams)*self.epsilon
    self.b = np.zeros([self.nParams,1])
    self.w = np.zeros([self.nParams], dtype=float)

class SemiGradientTDControl:

  def __init__(self, nParams, nActions, alpha, approximationFunctionArgs, actionSelectionMethod="egreedy", epsilon=0.01):
    self.name = "Generic SemiGradient TD Control Class"
    self.nParams = nParams
    self.nActions = nActions
    self.alpha = alpha
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.w = np.zeros([self.nParams], dtype=float)
    self.policy = FunctionApproximationPolicy(self.nParams, self.nActions, self.af_kwargs, 
      actionSelectionMethod=actionSelectionMethod, epsilon=epsilon)
  
  def selectAction(self, state):
    return self.policy.selectAction(state)
    
  def getValue(self, state, action=None):
    if action is None:
      return np.array([self.af(self.w, state, a, **self.af_kwargs) for a in range(self.nActions)])
    else:
      return self.af(self.w, state, action, **self.af_kwargs)
    
  def getName(self):
    return self.name
    
  def reset(self):
    self.w = np.zeros([self.nParams], dtype=float)
    self.policy.reset()    
 
  def getGreedyAction(self, state, actionsAvailable=None):
    q = np.array([self.af(self.w, state, a, **self.af_kwargs) for a in range(self.nActions)])
    if(actionsAvailable is None):
      actionValues = q[:]
      actionList = np.array(range(self.nActions))
    else:
      actionValues = q[actionsAvailable]
      actionList = np.array(actionsAvailable)
    actionIdx = selectAction_greedy(actionValues)
    return actionList[actionIdx]
    
class SemiGradientSARSA(SemiGradientTDControl):
  
  def __init__(self, nParams, nActions, alpha, gamma, approximationFunctionArgs, actionSelectionMethod="egreedy", epsilon=0.01):
    super().__init__(nParams, nActions, alpha, approximationFunctionArgs, actionSelectionMethod, epsilon)
    self.name = "Semi-gradient SARSA"
    self.gamma = gamma

  def update(self, episode):  
    t = len(episode)-2
    state = episode[t]["state"]
    action = episode[t]["action"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    next_action = episode[t+1]["action"]      
    done = episode[t+1]["done"]
    U = reward
    if not done:
      U += self.gamma * self.getValue(next_state, next_action)
    self.w += self.alpha*(U - self.getValue(state, action)) * self.afd(self.w, state, action, **self.af_kwargs)
    self.policy.update(self.w)

class SemiGradientExpectedSARSA(SemiGradientTDControl):
  
  def __init__(self, nParams, nActions, alpha, gamma, approximationFunctionArgs, actionSelectionMethod="egreedy", epsilon=0.01):
    super().__init__(nParams, nActions, alpha, approximationFunctionArgs, actionSelectionMethod, epsilon)
    self.name = "Semi-gradient Expected SARSA"
    self.gamma = gamma

  def update(self, episode):  
    t = len(episode)-2
    state = episode[t]["state"]
    action = episode[t]["action"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    done = episode[t+1]["done"]
    U = reward
    if not done:
      q_next = np.array([self.getValue(next_state, next_action) for next_action in range(self.nActions)])
      action_probs = self.policy.getProbability(next_state)
      U += self.gamma * np.dot(action_probs, q_next)
    self.w += self.alpha*(U - self.getValue(state, action)) * self.afd(self.w, state, action, **self.af_kwargs)
    self.policy.update(self.w)

class SemiGradientQLearning(SemiGradientTDControl):
  
  def __init__(self, nParams, nActions, alpha, gamma, approximationFunctionArgs, actionSelectionMethod="egreedy", epsilon=0.01):
    super().__init__(nParams, nActions, alpha, approximationFunctionArgs, actionSelectionMethod, epsilon)
    self.name = "Semi-gradient Q-Learning"
    self.gamma = gamma

  def update(self, episode):  
    t = len(episode)-2
    state = episode[t]["state"]
    action = episode[t]["action"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    done = episode[t+1]["done"]
    q_now = self.getValue(state, action)
    q_next = np.max(self.getValue(next_state))
    td_error = reward + self.gamma*q_next - q_now
    self.w += self.alpha * td_error * self.afd(self.w, state, action, **self.af_kwargs)
    self.policy.update(self.w)
    
class DifferentialSemiGradientSARSA(SemiGradientTDControl):
  
  def __init__(self, nParams, nActions, alpha, beta, approximationFunctionArgs, actionSelectionMethod="egreedy", epsilon=0.01):
    super().__init__(nParams, nActions, alpha, approximationFunctionArgs, actionSelectionMethod, epsilon)
    self.name = "Differential semi-gradient SARSA"
    self.beta = beta
    self.avgR = 0.0
  
  def update(self, episode):  
    t = len(episode)-2
    state = episode[t]["state"]
    action = episode[t]["action"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    next_action = episode[t+1]["action"]      
    done = episode[t+1]["done"]
    q_now = self.getValue(state, action)
    q_next = self.getValue(next_state, next_action)
    td_error = reward - self.avgR + q_next - q_now
    self.avgR += self.beta*td_error
    self.w += self.alpha * td_error * self.afd(self.w, state, action, **self.af_kwargs)
    self.policy.update(self.w)

  def reset(self):
    super().reset()
    self.avgR = 0.0

class DifferentialSemiGradientQLearning(SemiGradientTDControl):
  
  def __init__(self, nParams, nActions, alpha, beta, approximationFunctionArgs, actionSelectionMethod="egreedy", epsilon=0.01):
    super().__init__(nParams, nActions, alpha, approximationFunctionArgs, actionSelectionMethod, epsilon)
    self.name = "Differential semi-gradient Q-Learning"
    self.beta = beta
    self.avgR = 0.0

  def update(self, episode):  
    t = len(episode)-2
    state = episode[t]["state"]
    action = episode[t]["action"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    done = episode[t+1]["done"]
    q_now = self.getValue(state, action)
    q_next = np.max(self.getValue(next_state))
    td_error = reward - self.avgR + q_next - q_now
    self.avgR += self.beta*td_error
    self.w += self.alpha * td_error * self.afd(self.w, state, action, **self.af_kwargs)
    self.policy.update(self.w)

  def reset(self):
    super().reset()
    self.avgR = 0.0

class nStepSemiGradientTDControl(SemiGradientTDControl):

  def __init__(self, nParams, nActions, alpha, n, approximationFunctionArgs, actionSelectionMethod="egreedy", epsilon=0.01):
    super().__init__(nParams, nActions, alpha, approximationFunctionArgs, actionSelectionMethod, epsilon)
    self.name = "Generic n Step Semi-Gradient TD Control Class"
    self.n = n
    self.bufferExperience = []

  def update(self, episode):    
    self.updateBuffer(episode)
    t = len(self.bufferExperience)-2
    tau = t + 1 - self.n
    if tau>=0:
      self.sweepBuffer(tau, tau+1)
    if self.bufferExperience[t+1]['done']:
      self.bufferExperience = []
    self.policy.update(self.w)
    
  def updateBuffer(self, episode):
    self.bufferExperience = episode
    while(len(self.bufferExperience)>(self.n+1)):
      self.bufferExperience.pop(0)

  def reset(self):
    super().reset()
    self.bufferExperience = []

class nStepSemiGradientSARSA(nStepSemiGradientTDControl):

  def __init__(self, nParams, nActions, alpha, gamma, n, approximationFunctionArgs, actionSelectionMethod="egreedy", epsilon=0.01):
    super().__init__(nParams, nActions, alpha, n, approximationFunctionArgs, actionSelectionMethod, epsilon)
    self.name = "Semi-gradient n-step SARSA"
    self.gamma = gamma
  
  def sweepBuffer(self, tau_start, tau_stop):
    t = len(self.bufferExperience)-2
    for tau in range(tau_start, tau_stop):
      state = self.bufferExperience[tau]['state']
      action = self.bufferExperience[tau]['action']
      rewards = np.array([self.bufferExperience[i]['reward'] for i in range(tau+1, min(tau+self.n, t+1)+1)])
      gammas = np.array([self.gamma**i for i in range(min(self.n, t+1-tau))])
      if((tau+self.n)>t+1):
        G = np.sum(rewards * gammas)
      else:       
        state_last = self.bufferExperience[tau+self.n]['state']
        action_last = self.bufferExperience[tau+self.n]['action']
        G = np.sum(rewards * gammas) + self.gamma**(self.n) * self.getValue(state_last, action_last)
      self.w += self.alpha*(G - self.getValue(state, action)) * self.afd(self.w, state, action, **self.af_kwargs)

class nStepDifferentialSemiGradientSARSA(nStepSemiGradientTDControl):

  def __init__(self, nParams, nActions, alpha, beta, n, approximationFunctionArgs, actionSelectionMethod="egreedy", epsilon=0.01):
    super().__init__(nParams, nActions, alpha, n, approximationFunctionArgs, actionSelectionMethod, epsilon)
    self.name = "Differential semi-gradient n-step SARSA"
    self.beta = beta
    self.avgR = 0.0

  def sweepBuffer(self, tau_start, tau_stop):
    for tau in range(tau_start, tau_stop):
      state = self.bufferExperience[tau]['state']
      action = self.bufferExperience[tau]['action']
      last_state = self.bufferExperience[tau+self.n]['state']
      last_action = self.bufferExperience[tau+self.n]['action']
      rewards = np.array([self.bufferExperience[i]['reward'] for i in range(tau+1, tau+self.n+1)])
      q_now = self.getValue(state, action)
      q_last = self.getValue(last_state, last_action)     
      td_error = np.sum(rewards - self.avgR) + q_last - q_now  
      self.avgR += self.beta*td_error
      self.w += self.alpha * td_error * self.afd(self.w, state, action, **self.af_kwargs)
  
  def reset(self):
    super().reset()
    self.avgR = 0.0

class nStepSemiGradientTreeBackup(nStepSemiGradientTDControl):

  def __init__(self, nParams, nActions, alpha, gamma, n, approximationFunctionArgs, actionSelectionMethod="egreedy", epsilon=0.01):
    super().__init__(nParams, nActions, alpha, n, approximationFunctionArgs, actionSelectionMethod, epsilon)
    self.name = "Differential semi-gradient n-step Tree Backup"
    self.gamma = gamma

  def sweepBuffer(self, tau_start, tau_stop):
    for tau in range(tau_start, tau_stop):
      state = self.bufferExperience[tau]['state']
      action = self.bufferExperience[tau]['action']
      q = self.getValue(state, action)
      cum_td_error = 0.0
      for k in range(tau, tau+self.n):
        state_k = self.bufferExperience[k]['state']
        action_k = self.bufferExperience[k]['action']
        reward = self.bufferExperience[k+1]['reward']
        next_state = self.bufferExperience[k+1]['state']
        q_next = self.getValue(next_state)
        action_probs = self.policy.getProbability(next_state)
        td_error = reward + self.gamma * np.dot(action_probs, q_next) - self.getValue(state_k, action_k)
        aux = [self.gamma * self.policy.getProbability(self.bufferExperience[i]['state'],self.bufferExperience[i]['action']) for i in range(tau+1,k+1)]
        cum_td_error += td_error * np.prod(aux)
      G = q + cum_td_error
      self.w += self.alpha * (G - q) * self.afd(self.w, state, action, **self.af_kwargs)
      self.policy.update(self.w)