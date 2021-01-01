'''
EligibilityTraces.py : Implementation of Eligibility Traces methods

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
from IRL.utils.Helpers import getValueFromDict
from IRL.utils.Policies import FunctionApproximationPolicy

class OfflineLambdaReturn:
  
  def __init__(self, nParams, alpha, gamma, lambd, approximationFunctionArgs):
    self.name = "Offline Lambda Return"
    self.nParams = nParams
    self.alpha = alpha
    self.gamma = gamma
    self.lambd = lambd
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.w = np.zeros([self.nParams], dtype=np.float)
    self.bufferExperience = []
  
  def evaluate(self, episode):
    self.updateBuffer(episode)
    if self.bufferExperience[-1]['done']:
      T = len(self.bufferExperience)-1
      for t in range(T):
        state = self.bufferExperience[t]['state']
        G = 0.0
        sumReturns = 0.0
        for n in range(1, T-t):
          G += self.gamma**(n-1)*self.bufferExperience[t+n]['reward']
          sumReturns += self.lambd**(n-1)*(G + (self.gamma**n) * self.getValue(self.bufferExperience[t+n]['state']))
        G_t = G + self.gamma**(T-t-1)*self.bufferExperience[T]['reward']
        G_lambda = (1.0 - self.lambd)*sumReturns + self.lambd**(T-t-1)*G_t
        self.w += self.alpha*(G_lambda - self.getValue(state))*self.afd(self.w, state, **self.af_kwargs)
      self.resetBuffer()
      
  def updateBuffer(self, episode):
    self.bufferExperience = episode
    
  def resetBuffer(self):
    self.bufferExperience = []
    
  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
  
  def reset(self):
    self.w = np.zeros([self.nParams], dtype=np.float)
    self.bufferExperience = []
    
  def getName(self):
    return self.name
  
class OnlineLambdaReturn:
  
  def __init__(self, nParams, alpha, gamma, lambd, approximationFunctionArgs):
    self.name = "Online Lambda Return"
    self.nParams = nParams
    self.alpha = alpha
    self.gamma = gamma
    self.lambd = lambd
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")  
    self.w = np.zeros([self.nParams], dtype=np.float)
    self.bufferExperience = []
  
  def evaluate(self, episode):
    self.updateBuffer(episode)
    T = len(self.bufferExperience)-1
    for t in range(T):
      state = self.bufferExperience[t]['state']
      G = 0.0
      sumReturns = 0.0
      for n in range(1, T-t):
        G += self.gamma**(n-1)*self.bufferExperience[t+n]['reward']
        sumReturns += self.lambd**(n-1)*(G + (self.gamma**n) * self.getValue(self.bufferExperience[t+n]['state']))
      G_t = G + self.gamma**(T-t-1)*self.bufferExperience[T]['reward']
      G_lambda = (1.0 - self.lambd)*sumReturns + self.lambd**(T-t-1)*G_t
      self.w += self.alpha*(G_lambda - self.getValue(state))*self.afd(self.w, state, **self.af_kwargs)
    if self.bufferExperience[-1]['done']:
      self.resetBuffer()
      
  def updateBuffer(self, episode):
    self.bufferExperience = episode
    
  def resetBuffer(self):
    self.bufferExperience = []
    
  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
  
  def reset(self):
    self.w = np.zeros([self.nParams], dtype=np.float)
    self.bufferExperience = []
    
  def getName(self):
    return self.name
    
class SemiGradientTDLambda:
  
  def __init__(self, nParams, alpha, gamma, lambd, approximationFunctionArgs):
    self.name = "Semi-Gradient TD(Lambda)"
    self.nParams = nParams
    self.alpha = alpha
    self.gamma = gamma
    self.lambd = lambd
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.w = np.zeros([self.nParams], dtype=np.float)
    self.z = np.zeros([self.nParams], dtype=np.float)
  
  def evaluate(self, episode):
    t = len(episode)-2
    state = episode[t]["state"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    done = episode[t+1]["done"]     
    self.z = self.gamma*self.lambd*self.z + self.afd(self.w, state, **self.af_kwargs)
    td_error = reward + self.gamma*self.getValue(next_state) - self.getValue(state)
    self.w += self.alpha*td_error*self.z
    if(done):
      self.z *= 0.0 
    
  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)
    
  def reset(self):
    self.w = np.zeros([self.nParams], dtype=np.float)
    self.z = np.zeros([self.nParams], dtype=np.float)

  def getName(self):
    return self.name
    
class TrueOnlineTDLambda:
  
  def __init__(self, nParams, alpha, gamma, lambd, approximationFunctionArgs):
    self.name = "True online TD(Lambda)"
    self.nParams = nParams
    self.alpha = alpha
    self.gamma = gamma
    self.lambd = lambd
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.afd = getValueFromDict(self.af_kwargs, "afd")
    self.ftf = getValueFromDict(self.af_kwargs, "ftf")
    self.w = np.zeros([self.nParams], dtype=np.float)
    self.z = np.zeros([self.nParams], dtype=np.float)
    self.v_old = 0.0
  
  def evaluate(self, episode):
    t = len(episode)-2
    state = episode[t]["state"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    done = episode[t+1]["done"]
    x = self.ftf(state, **self.af_kwargs)
    v = self.getValue(state)
    v_next = self.getValue(next_state)
    td_error = reward + self.gamma*v_next - v
    self.z = self.gamma*self.lambd*self.z + (1.0-self.alpha*self.gamma*self.lambd*np.dot(self.z,x))*x
    self.w += self.alpha*(td_error+v-self.v_old)*self.z - self.alpha*(v-self.v_old)*x
    self.v_old = v_next
    if(done):
      self.z *= 0.0
      self.v_old = 0.0
    
  def getValue(self, state):
    return self.af(self.w, state, **self.af_kwargs)

  def reset(self):
    self.w = np.zeros([self.nParams], dtype=np.float)
    self.z = np.zeros([self.nParams], dtype=np.float)
    self.v_old = 0.0
    
  def getName(self):
    return self.name
    
class SARSALambda:
  
  def __init__(self, nParams, nActions, alpha, gamma, lambd, approximationFunctionArgs, 
    doAccumulateTraces=False, doClearTraces=False, actionSelectionMethod="egreedy", epsilon=0.01):
    self.name = "SARSA(Lambda)"
    self.nParams = nParams
    self.nActions = nActions
    self.alpha = alpha
    self.gamma = gamma
    self.lambd = lambd
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.ftf = getValueFromDict(self.af_kwargs, "ftf")
    self.doAccumulateTraces = doAccumulateTraces
    self.doClearTraces = doClearTraces
    self.w = np.zeros([self.nParams], dtype=np.float)
    self.z = np.zeros([self.nParams], dtype=np.float)
    self.policy = FunctionApproximationPolicy(self.nParams, self.nActions, self.af_kwargs, 
      actionSelectionMethod=actionSelectionMethod, epsilon=epsilon)
  
  def update(self, episode):
    t = len(episode)-2
    state = episode[t]["state"]
    action = episode[t]["action"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    next_action = episode[t+1]["action"]
    done = episode[t+1]["done"]
    x = self.ftf(state, action, **self.af_kwargs)
    xx = self.ftf(next_state, next_action, **self.af_kwargs)
    td_error = reward
    for i in np.nonzero(x)[0]:
      td_error-=self.w[i]
      if self.doAccumulateTraces:
        self.z[i]+=1
      else:
        self.z[i]=1
    if done:
      self.w += self.alpha*td_error*self.z
      self.policy.update(self.w)
      self.z *= 0.0
    else:
      for i in np.nonzero(xx)[0]:
        td_error+=self.gamma*self.w[i]    
      self.w += self.alpha*td_error*self.z
      self.policy.update(self.w)
      self.z = self.gamma*self.lambd*self.z
    if self.doClearTraces:
      idxToClear = np.array(np.ones(self.nParams), dtype=int)
      idxToClear[np.nonzero(x)[0]] = 0
      idxToClear[np.nonzero(xx)[0]] = 0
      self.z[idxToClear] = 0.0
 
  def getValue(self, state, action=None):
    if action is None:
      return np.array([self.af(self.w, state, action, **self.af_kwargs) for action in range(self.nActions)])
    else:
      return self.af(self.w, state, action, **self.af_kwargs)

  def selectAction(self, state):
    return self.policy.selectAction(state)
    
  def reset(self):
    self.w = np.zeros([self.nParams], dtype=np.float)
    self.z = np.zeros([self.nParams], dtype=np.float)
    
  def getName(self):
    return self.name
    
class TrueOnlineSARSA:
  
  def __init__(self, nParams, nActions, alpha, gamma, lambd, approximationFunctionArgs, actionSelectionMethod="egreedy", epsilon=0.01):
    self.name = "True Online SARSA"
    self.nParams = nParams
    self.nActions = nActions
    self.alpha = alpha
    self.gamma = gamma
    self.lambd = lambd
    self.af_kwargs = approximationFunctionArgs
    self.af = getValueFromDict(self.af_kwargs, "af")
    self.ftf = getValueFromDict(self.af_kwargs, "ftf")
    self.w = np.zeros([self.nParams], dtype=np.float)
    self.z = np.zeros([self.nParams], dtype=np.float)
    self.q_old = 0.0
    self.policy = FunctionApproximationPolicy(self.nParams, self.nActions, self.af_kwargs, 
      actionSelectionMethod=actionSelectionMethod, epsilon=epsilon)
  
  def update(self, episode):
    t = len(episode)-2
    state = episode[t]["state"]
    action = episode[t]["action"]
    reward = episode[t+1]["reward"]
    next_state = episode[t+1]["state"]      
    next_action = episode[t+1]["action"]
    done = episode[t+1]["done"]
    x = self.ftf(state, action, **self.af_kwargs)
    xx = self.ftf(next_state, next_action, **self.af_kwargs)
    q = self.getValue(state, action)
    q_next = self.getValue(next_state, next_action)
    td_error = reward + self.gamma*q_next - q
    self.z = self.gamma*self.lambd*self.z + (1-self.alpha*self.gamma*self.lambd*np.dot(self.z, x))*x
    self.w += self.alpha * (td_error + q - self.q_old)*self.z - self.alpha*(q - self.q_old)*x
    self.policy.update(self.w)
    self.q_old = q_next
    if done:
      self.z *= 0.0
      self.q_old = 0.0
    
  def getValue(self, state, action=None):
    if action is None:
      return np.array([self.af(self.w, state, action, **self.af_kwargs) for action in range(self.nActions)])
    else:
      return self.af(self.w, state, action, **self.af_kwargs)
  
  def selectAction(self, state):
    return self.policy.selectAction(state)
    
  def reset(self):
    self.w = np.zeros([self.nParams], dtype=np.float)
    self.z = np.zeros([self.nParams], dtype=np.float)
    self.q_old = 0.0
    
  def getName(self):
    return self.name