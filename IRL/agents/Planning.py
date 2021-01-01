'''
Planning.py : Implementations of models for planning based agents

Cem Karaoguz, 2020
MIT License
'''

import numpy as np

class DeterministicModel():

  MAX_TRANSITIONBOOKKEEPER = 1000000

  def __init__(self, nStates, nActions, kappa=0.0, doExtendActions=False):
    self.nStates = nStates
    self.nActions = nActions
    self.kappa = kappa
    self.doExtendActions = doExtendActions
    self.availableStates = np.zeros([self.nStates, nActions], dtype=int)
    self.stateTransitionModel = np.zeros([self.nStates, self.nActions], dtype=int)
    for idx_state in range(self.nStates):
      self.stateTransitionModel[idx_state,:] = idx_state
    self.rewardModel = np.zeros([self.nStates, self.nActions], dtype=float)
    self.transitionBookkeeper = np.zeros([self.nStates, self.nActions], dtype=int)
    self.terminalStates = []
    
  def __getitem__(self, idx):
    state, action = idx
    return (self.rewardModel[state, action], self.stateTransitionModel[state, action])
      
  def update(self, experiences):
    T = len(experiences)
    for t in range(T-2, T-1):
      state = experiences[t]["state"]
      action = experiences[t]["action"]
      reward = experiences[t+1]["reward"]
      next_state = experiences[t+1]["state"]
      done = experiences[t+1]["done"]
      if(self.kappa>0 or self.doExtendActions):
        # DynaQ+
        self.availableStates[state, :] = 1
      else:
        self.availableStates[state, action] = 1
      self.stateTransitionModel[state, action] = next_state
      self.rewardModel[state, action] = reward
      if(done and next_state not in self.terminalStates):
        self.terminalStates.append(next_state)
      self.transitionBookkeeper += 1
      self.transitionBookkeeper[state, action] = 0
      if(np.max(self.transitionBookkeeper[:])>=self.MAX_TRANSITIONBOOKKEEPER):
        self.transitionBookkeeper = np.zeros([self.nSTates, self.nActions])
    
  def step(self, state, action):
    next_state = None
    reward = None
    done = None
    if(self.availableStates[state,action]==1):
      next_state = self.stateTransitionModel[state, action]
      reward = self.rewardModel[state, action]
      if(next_state in self.terminalStates):
        done = True
      else:
        done = False
    return (next_state, reward, done)
    
  def sampleExperience(self):
    state = np.random.choice(np.nonzero(np.max(self.availableStates,1))[0])
    action = np.random.choice(np.nonzero(self.availableStates[state,:])[0])
    next_state = self.stateTransitionModel[state, action]
    reward = self.rewardModel[state, action] + self.kappa*np.sqrt(self.transitionBookkeeper[state, action])
    experiences = []
    xp0 = {}
    xp0['state'] = state
    xp0['action'] = action
    xp0['done'] = state in self.terminalStates
    experiences.append(xp0)
    xp1 = {}
    xp1['reward'] = reward
    xp1['state'] = next_state
    xp1['done'] = next_state in self.terminalStates
    experiences.append(xp1)
    return experiences
    
  def getAncestors(self, state):
    ret = []
    transitions_ancestor = np.array(np.nonzero(self.stateTransitionModel==state)).T
    for tr in transitions_ancestor:
      ancestor_state = tr[0]
      ancestor_action = tr[1]
      if self.availableStates[ancestor_state, ancestor_action]:
        ret.append(np.array([ancestor_state,ancestor_action,self.rewardModel[ancestor_state,ancestor_action]]))
    return ret
