'''
ToyExamples.py : Implementation of various toy examples presented in the book

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
from IRL.utils.Helpers import getValueFromDict

class RandomWalk():

  ACTION_LEFT = 0
  ACTION_RIGHT = 1
  
  def __init__(self, nStatesOneSide, pLeft=0.5, defaultReward=0.0, specialRewards={}):
    # States
    self.nStates = (int)(nStatesOneSide*2 + 1)
    self.startState = (int)(nStatesOneSide)
    self.terminalStates = [0, self.nStates-1]
    # Actions
    self.pLeft = pLeft
    self.actionMapping = {self.ACTION_LEFT:(-1, "Left"), self.ACTION_RIGHT:(1, "Right")}
    # Rewards
    self.defaultReward = defaultReward
    self.specialRewards = specialRewards
    self.agentState = self.startState
    self.lastAction = None
    
  def step(self):
    coin_toss = np.random.binomial(1, self.pLeft)
    if(coin_toss==1):
      # Left
      self.lastAction = self.ACTION_LEFT
    else:
      # Right
      self.lastAction = self.ACTION_RIGHT
    self.agentState = self.agentState + self.actionMapping[self.lastAction][0]
    if(self.agentState in self.specialRewards):
      reward = self.specialRewards[self.agentState]
    else:
      reward = self.defaultReward
    if(self.agentState in self.terminalStates):
      done = True
    else:
      done = False
    return self.agentState, reward, done
    
  def reset(self):
    self.agentState = self.startState
    self.lastAction = None
    return self.agentState
    
  def printEnv(self):
    pass
    
class InfiniteVariance():

  N_STATES = 2
  STATE_S = 0
  STATE_TERMINAL = 1
  N_ACTIONS = 2
  ACTION_LEFT = 0
  ACTION_RIGHT = 1
  
  def __init__(self, p_01=0.1):
    # States
    self.nStates = self.N_STATES
    self.startState = self.STATE_S
    self.terminalStates = [self.STATE_TERMINAL]
    # Actions
    self.nActions = self.N_ACTIONS
    # State transitions
    self.p_01 = p_01
    # Rewards
    self.rewardFunction = np.zeros([self.nStates, self.nActions, self.nStates])
    self.rewardFunction[0,self.ACTION_LEFT,1] = 1
    self.agentState = self.startState
    
  def step(self, action):
    if(self.agentState==self.STATE_S):
      # 0: s
      if(action==self.ACTION_LEFT):
        # Left
        if(np.random.rand()<self.p_01):
          next_state = self.STATE_TERMINAL
          done = True
        else:
          next_state = self.STATE_S
          done = False
      else:
        # Right
        next_state = self.STATE_TERMINAL
        done = True
    elif(self.agentState==self.STATE_TERMINAL):
      # 1: Terminal
      next_state = self.STATE_TERMINAL
      done = True
    else:
      # Not supposed to be here!
      return None
    reward = self.rewardFunction[self.agentState, action, next_state]
    self.agentState = next_state
    
    return next_state, reward, done
    
  def reset(self):
    self.agentState = self.startState
    return self.agentState
    
  def printEnv(self):
    pass

class MaximizationBias():

  N_STATES = 4
  STATE_B = 1
  STATE_A = 2
  STATE_TERMINAL_R = 3
  STATE_TERMINAL_L = 0
  N_ACTIONS_MIN = 2
  ACTION_LEFT = 1
  ACTION_RIGHT = 0
  
  def __init__(self, mu=-0.1, std=1, nActions=10):
    self.mu = mu
    self.std = std
    #self.nActions = max(self.N_ACTIONS_MIN, nActions)
    self.nActions = self.N_ACTIONS_MIN + nActions
    # States
    self.nStates = self.N_STATES
    self.startState = self.STATE_A
    self.terminalStates = [self.STATE_TERMINAL_L, self.STATE_TERMINAL_R]
    self.stateTransitions = {
      self.STATE_A: {self.ACTION_RIGHT: self.STATE_TERMINAL_R, self.ACTION_LEFT: self.STATE_B},
      self.STATE_B: {a: self.STATE_TERMINAL_L for a in range(self.N_ACTIONS_MIN, self.N_ACTIONS_MIN+self.nActions)},
      self.STATE_TERMINAL_R: {self.ACTION_RIGHT: self.STATE_TERMINAL_R, self.ACTION_LEFT: self.STATE_TERMINAL_R},
      self.STATE_TERMINAL_L: {self.ACTION_LEFT: self.STATE_TERMINAL_L, self.ACTION_RIGHT: self.STATE_TERMINAL_L}
    }
    # Actions
    self.stateActionMapping = {
      self.STATE_A: [self.ACTION_RIGHT, self.ACTION_LEFT],
      #self.STATE_B: [i for i in range(self.nActions)],
      self.STATE_B: [i for i in range(self.N_ACTIONS_MIN,self.N_ACTIONS_MIN+nActions)],
      self.STATE_TERMINAL_R: [self.ACTION_RIGHT], 
      self.STATE_TERMINAL_L: [self.ACTION_LEFT] }
    # Rewards
    self.defaultReward = 0.0
    self.agentState = self.startState
    
  def step(self, action):
    self.agentState = self.stateTransitions[self.agentState][action]
    if(self.agentState == self.STATE_TERMINAL_L):
      reward = np.random.normal(self.mu, self.std)
    else:
      reward = self.defaultReward
    if(self.agentState in self.terminalStates):
      done = True
    else:
      done = False
    return self.agentState, reward, done
    
  def reset(self):
    self.agentState = self.startState
    return self.agentState
    
  def getAvailableActions(self, state=None):
    if state is None:
      return self.stateActionMapping[self.agentState]
    else:
      return self.stateActionMapping[state]
  
  def printEnv(self):
    pass
    
class TrajectorySamplingTask():

  STATE_START = 0
  ACTION_LEFT = 0
  ACTION_RIGHT = 1
  REWARD_TERMINAL = 0.0

  def __init__(self, nStates, b, pTerminal=0.1):
    self.nStates = nStates
    self.b = b
    self.pTerminal = pTerminal
    self.startState = self.STATE_START
    self.actionMapping = {self.ACTION_LEFT:(0, "Left"), self.ACTION_RIGHT:(1, "Right")}   
    self.nActions = len(self.actionMapping) 
    self.stateTransitions = np.random.randint(self.nStates, size=(self.nStates, self.nActions, self.b))
    self.rewardFunction = np.random.randn(self.nStates, self.nActions, self.b)
    self.currentState = self.STATE_START

  def step(self, action):
      if np.random.rand() < self.pTerminal:
        next_state = self.nStates
        reward = self.REWARD_TERMINAL
        done = True
      else:
        next = np.random.randint(self.b)
        next_state = self.stateTransitions[self.currentState, action, next]
        reward = self.rewardFunction[self.currentState, action, next]
        done = False
      self.currentState = next_state
      return next_state, reward, done
      
  def reset(self):
    self.currentState = self.STATE_START
    return self.currentState
    
  def computeExpectedValue(self, state, action, valueTable, gamma):
    next_states = self.stateTransitions[state, action]
    expected_rewards = self.rewardFunction[state, action]
    return ((1.0 - self.pTerminal) / self.b) * sum(expected_rewards[i] + valueTable[s_p] for i, s_p in enumerate(next_states))

  def computeExpectedUpdate(self, state, action, actionValueTable):
    next_states = self.stateTransitions[state, action]
    expected_rewards = self.rewardFunction[state, action]
    return (1.0 - self.pTerminal) * np.mean(expected_rewards + np.max(actionValueTable[next_states, :], axis=1))

class BairdsCounterExample:

  N_STATES = 7
  N_ACTIONS = 2
  ACTION_DASHED = 0
  ACTION_SOLID = 1
  DEFAULT_REWARD = 0.0
  
  def __init__(self):
    self.nStates = self.N_STATES
    self.nActions = self.N_ACTIONS
    self.defaultReward = self.DEFAULT_REWARD
    self.actionMapping = {self.ACTION_IDX_DASHED:[0, "Dashed"], self.ACTION_SOLID:[1, "Solid"]}
    self.stateTransitionProbs = np.zeros([self.nStates, self.nActions, self.nStates])
    self.stateTransitionProbs[:,self.ACTION_DASHED,:] = 1.0/self.nStates
    self.stateTransitionProbs[:,self.ACTION_SOLID,6] = 1.0
    self.state = None
    self.reset()
    
  def step(self, action):
    done = False
    next_state = np.random.choice(self.nStates, p=self.stateTransitionProbs[self.state,action,:])
    reward = self.defaultReward
    self.state = next_state
    return next_state, reward, done
    
  def reset(self):
    self.state = np.random.choice(self.nStates)
    return self.state
    
  def computeExpectedValue(self, state, action, w, af_kwargs, gamma):
    af = getValueFromDict(af_kwargs, "af")
    return np.sum( [self.stateTransitionProbs[state, action, next_state] * (self.defaultReward + gamma*af(w, next_state, **af_kwargs)) for next_state in range(self.nStates)] )
    
class ShortCorridor:

  N_STATES = 4
  N_ACTIONS = 2
  STATE_START = 0
  STATE_TERMINAL = 3
  ACTION_LEFT = 0
  ACTION_RIGHT = 1
  DEFAULT_REWARD = -1.0
  
  def __init__(self):
    self.nStates = self.N_STATES
    self.nActions = self.N_ACTIONS
    self.defaultReward = self.DEFAULT_REWARD
    self.stateTransitionProbs = np.zeros([self.nStates, self.nActions, self.nStates])
    self.stateTransitionProbs[0,self.ACTION_LEFT,0] = 1.0
    self.stateTransitionProbs[0,self.ACTION_RIGHT,1] = 1.0
    self.stateTransitionProbs[1,self.ACTION_LEFT,2] = 1.0
    self.stateTransitionProbs[1,self.ACTION_RIGHT,0] = 1.0
    self.stateTransitionProbs[2,self.ACTION_LEFT,1] = 1.0
    self.stateTransitionProbs[2,self.ACTION_RIGHT,3] = 1.0
    self.stateTransitionProbs[3,:,3] = 1.0
    self.state = None
    self.reset()
    
  def step(self, action):
    if self.state==self.STATE_TERMINAL:
      return self.state, self.defaultReward, True
    next_state = np.random.choice(self.nStates, p=self.stateTransitionProbs[self.state,action,:])
    reward = self.defaultReward
    self.state = next_state
    if next_state==self.STATE_TERMINAL:
      done = True
    else:
        done = False
    return next_state, reward, done
    
  def reset(self):
    self.state = self.STATE_START
    return self.state
    