'''
PolicyGradient.py : Policy gradient algorithms

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
from IRL.utils.Policies import StochasticPolicy, ParametrizedPolicy
from IRL.utils.Helpers import getValueFromDict

class BanditGradient():

  def __init__(self, nStates, nActions, alpha, doUseBaseline=True):
    self.nStates = nStates
    self.nActions = nActions
    self.alpha = alpha
    self.doUseBaseline = doUseBaseline
    self.preferencesTable = np.zeros([self.nStates, self.nActions], dtype=float)+0.0001
    self.policy = StochasticPolicy(self.nStates, self.nActions, policyUpdateMethod="softmax", tieBreakingMethod="consistent")
    self.count = 0
    self.avgReward = 0.0
    
  def update(self, state, action, reward):
    if self.doUseBaseline:
      baseline = self.avgReward
    else:
      baseline = 0.0
    for a in range(self.nActions):
      if(a==action):
        self.preferencesTable[state, a] += self.alpha*(reward - baseline)*(1.0 - self.policy.getProbability(state,a))
      else:
        self.preferencesTable[state, a] -= self.alpha*(reward - baseline)*self.policy.getProbability(state,a)
    self.policy.update(state, self.preferencesTable)
    self.count += 1
    self.avgReward = self.avgReward + (1.0/self.count)*(reward - self.avgReward)
    
  def selectAction(self, state):
    return self.policy.sampleAction(state)
    
  def reset(self):
    self.preferencesTable = np.zeros([self.nStates, self.nActions], dtype=float)+0.0001
    self.count = 0
    self.avgReward = 0.0

class REINFORCE:

  def __init__(self, alpha, gamma, nParams, nActions, policyApproximationFunctionArgs):
    self.name = "REINFORCE"
    self.alpha = alpha
    self.gamma = gamma
    self.policy = ParametrizedPolicy(nParams, nActions, policyApproximationFunctionArgs)
    self.bufferExperience = []
    
  def update(self, episode):
    self.updateBuffer(episode)
    T = len(self.bufferExperience)-1
    if self.bufferExperience[T]["done"]==True:
      rewards = np.array([self.bufferExperience[i]["reward"] for i in range(1, T+1)])
      gammas = np.array([self.gamma**i for i in range(0, T)])
      for t in range(T):
        state = self.bufferExperience[t]["state"]
        action = self.bufferExperience[t]["action"]
        G = np.dot(rewards[t:T+1], gammas[0:T-t])
        dPolicy = self.policy.grad(state, action)
        self.policy.theta += self.alpha * self.gamma**t * G * dPolicy
      self.bufferExperience = []
  
  def updateBuffer(self, episode):
    self.bufferExperience = episode
  
  def selectAction(self, state):
    return self.policy.sampleAction(state)
    
  def reset(self):
    self.bufferExperience = []
    self.policy.reset()
  
  def getName(self):
    return self.name

  def getGreedyAction(self, state, availableActions=None):
    return self.selectAction(state)
    
class REINFORCEwithBaseline:

  def __init__(self, alpha_w, alpha_theta, gamma, nParams_w, approximationFunctionArgs, nParams_theta, nActions, policyApproximationFunctionArgs):
    self.name = "REINFORCE with Baseline"
    self.alpha_w = alpha_w
    self.alpha_theta = alpha_theta
    self.gamma = gamma
    self.nParams_w = nParams_w
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.w = np.zeros([self.nParams_w], dtype=np.float)
    self.policy = ParametrizedPolicy(nParams_theta, nActions, policyApproximationFunctionArgs)
    self.bufferExperience = []
    
  def update(self, episode):
    self.updateBuffer(episode)
    T = len(self.bufferExperience)-1
    if self.bufferExperience[T]["done"]==True:
      rewards = np.array([self.bufferExperience[i]["reward"] for i in range(1, T+1)])
      gammas = np.array([self.gamma**i for i in range(0, T)])
      for t in range(T):
        state = self.bufferExperience[t]["state"]
        action = self.bufferExperience[t]["action"]
        G = np.dot(rewards[t:T+1], gammas[0:T-t])
        td_error = G - self.getValue(state)
        dPolicy = self.policy.grad(state, action)
        self.w += self.alpha_w * td_error * self.afd(self.w, state, **self.af_kwargs)
        self.policy.theta += self.alpha_theta * self.gamma**t * td_error * dPolicy
      self.bufferExperience = []
        
  def updateBuffer(self, episode):
    self.bufferExperience = episode
  
  def selectAction(self, state):
    return self.policy.sampleAction(state)

  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
      
  def reset(self):
    self.bufferExperience = []
    self.w[:] = 0.0
    self.policy.reset()
  
  def getName(self):
    return self.name
    
  def getGreedyAction(self, state, availableActions=None):
    return self.selectAction(state)
    
class OneStepActorCritic:

  def __init__(self, alpha_w, alpha_theta, gamma, nParams_w, approximationFunctionArgs, nParams_theta, nActions, policyApproximationFunctionArgs):
    self.name = "One step Actor-Critic"
    self.alpha_w = alpha_w
    self.alpha_theta = alpha_theta
    self.gamma = gamma
    self.nParams_w = nParams_w
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.w = np.zeros([self.nParams_w], dtype=np.float)
    self.I = 1.0
    self.policy = ParametrizedPolicy(nParams_theta, nActions, policyApproximationFunctionArgs)
    
  def update(self, episode):
    t = len(episode)-2
    state = episode[t]["state"]
    action = episode[t]["action"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]
    done = episode[t+1]["done"]
    if done:
      v_next = 0
    else:
      v_next = self.getValue(next_state)
    td_error = reward + self.gamma * v_next - self.getValue(state)
    dPolicy = self.policy.grad(state, action)
    self.w += self.alpha_w * td_error * self.afd(self.w, state, **self.af_kwargs)
    self.policy.theta += self.alpha_theta * self.I * td_error * dPolicy
    if done:
      self.I = 1.0
    else:
      self.I*=self.gamma
    
  def selectAction(self, state):
    return self.policy.sampleAction(state)

  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
      
  def reset(self):
    self.w[:] = 0.0
    self.I = 1.0
    self.policy.reset()
  
  def getName(self):
    return self.name

  def getGreedyAction(self, state, availableActions=None):
    return self.selectAction(state)
    
class ActorCriticWithEligibilityTraces:

  def __init__(self, alpha_w, alpha_theta, gamma, lambd_w, lambd_theta, nParams_w, approximationFunctionArgs, 
    nParams_theta, nActions, policyApproximationFunctionArgs):
    self.name = "Actor-Criticwith Eligibility Traces"
    self.alpha_w = alpha_w
    self.alpha_theta = alpha_theta
    self.gamma = gamma
    self.lambd_w = lambd_w
    self.lambd_theta = lambd_theta
    self.nParams_w = nParams_w
    self.nParams_theta = nParams_theta
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.w = np.zeros([self.nParams_w], dtype=np.float)
    self.z_w = np.zeros([self.nParams_w], dtype=np.float)
    self.z_theta = np.zeros([self.nParams_theta], dtype=np.float)
    self.I = 1.0
    self.policy = ParametrizedPolicy(nParams_theta, nActions, policyApproximationFunctionArgs)
    
  def update(self, episode):
    t = len(episode)-2
    state = episode[t]["state"]
    action = episode[t]["action"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]
    done = episode[t+1]["done"]
    if done:
      v_next = 0
    else:
      v_next = self.getValue(next_state)
    td_error = reward + self.gamma * v_next - self.getValue(state)
    dPolicy = self.policy.grad(state, action)
    self.z_w = self.gamma * self.lambd_w * self.z_w + self.afd(self.w, state, **self.af_kwargs)
    self.z_theta = self.gamma * self.lambd_theta * self.z_theta + self.I * dPolicy
    self.w += self.alpha_w * td_error * self.z_w
    self.policy.theta += self.alpha_theta * td_error * self.z_theta
    if done:
      self.I = 1.0
      self.z_w = np.zeros([self.nParams_w], dtype=np.float)
      self.z_theta = np.zeros([self.nParams_theta], dtype=np.float)
    else:
      self.I*=self.gamma
    
  def selectAction(self, state):
    return self.policy.sampleAction(state)

  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
      
  def reset(self):
    self.w[:] = 0.0
    self.z_w[:] = 0.0
    self.z_theta[:] = 0.0
    self.I = 1.0
    self.policy.reset()
  
  def getName(self):
    return self.name

  def getGreedyAction(self, state, availableActions=None):
    return self.selectAction(state)
    
class ActorCriticWithEligibilityTracesAvgReward:

  def __init__(self, alpha_w, alpha_theta, alpha_r, lambd_w, lambd_theta, nParams_w, approximationFunctionArgs, 
    nParams_theta, nActions, policyApproximationFunctionArgs):
    self.name = "Actor-Criticwith Eligibility Traces (average reward)"
    self.alpha_w = alpha_w
    self.alpha_theta = alpha_theta
    self.alpha_r = alpha_r
    self.lambd_w = lambd_w
    self.lambd_theta = lambd_theta
    self.nParams_w = nParams_w
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.w = np.zeros([self.nParams_w], dtype=np.float)
    self.z_w = np.zeros([self.nParams_w], dtype=np.float)
    self.z_theta = np.zeros([self.nParams_theta], dtype=np.float)
    self.avgR = 0.0
    self.policy = ParametrizedPolicy(nParams_theta, nActions, policyApproximationFunctionArgs)
    
  def update(self, episode):
    t = len(episode)-2
    state = episode[t]["state"]
    action = episode[t]["action"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]
    td_error = reward - self.avgR + self.getValue(next_state) - self.getValue(state)
    self.avgR += self.alpha_r * td_error 
    dPolicy = self.policy.grad(state, action)
    self.z_w = self.lambd_w * self.z_w + self.afd(self.w, state)
    self.z_theta = self.lambd_theta * self.z_theta + dPolicy
    self.w += self.alpha_w * td_error * self.z_w
    self.policy.theta += self.alpha_theta * td_error * self.z_theta
    
  def selectAction(self, state):
    return self.policy.sampleAction(state)

  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
      
  def reset(self):
    self.w[:] = 0.0
    self.z_w[:] = 0.0
    self.z_theta[:] = 0.0
    self.avgR = 0.0
    self.policy.reset()
  
  def getName(self):
    return self.name
  
  def getGreedyAction(self, state, availableActions=None):
    return self.selectAction(state)