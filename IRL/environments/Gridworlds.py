'''
Gridworlds.py : implementations of gridworld based environments

Cem Karaoguz, 2020
MIT License
'''

import numpy as np

class GridWorld:

  VIS_NCHAR_PER_CELL = 3
  
  def __init__(self, sizeX, sizeY, startStates=[], terminalStates=[], impassableStates=[], specialStateTransitions={},
    defaultReward=-1.0, crashReward=None, finishReward=None, outOfGridReward=None, specialRewards={}):
    # States
    self.sizeX = sizeX
    self.sizeY = sizeY
    self.nStates = self.sizeX*self.sizeY    
    self.startStates = [self.getLinearIndex(s[0], s[1]) for s in startStates]
    self.terminalStates = [self.getLinearIndex(s[0], s[1]) for s in terminalStates]
    self.impassableStates = [self.getLinearIndex(s[0], s[1])for s in impassableStates]
    self.specialStateTransitions = {self.getLinearIndex(k[0], k[1]):self.getLinearIndex(v[0], v[1]) for k, v in specialStateTransitions.items()}
    # Actions
    self.actionMapping = {0:(np.array([0,-1]), "N"), 1:(np.array([0,1]), "S"), 2:(np.array([1,0]), "E"), 3:(np.array([-1,0]), "W")}   
    self.nActions = len(self.actionMapping)
    # Rewards
    self.defaultReward = defaultReward
    self.finishReward = finishReward if finishReward is not None else defaultReward
    self.outOfGridReward = outOfGridReward if outOfGridReward is not None else defaultReward
    self.crashReward = crashReward if crashReward is not None else defaultReward
    self.specialRewards = {(self.getLinearIndex(k[0][0],k[0][1]),k[1]):v for k, v in specialRewards.items()}
    # Agent state
    self.agentState = None
    self.reset()
    
  def step(self):
    agentPose = self.getCartesianIndex(self.agentState)    
    actionOnGrid = self.actionMapping[action][0]
    newPoseX = agentPose[0] + actionOnGrid[0]
    newPoseY = agentPose[1] + actionOnGrid[1]
    if(newPoseX<0 or newPoseX>=self.sizeX or newPoseY<0 or newPoseY>=self.sizeY):
      # Out of grid
      newPoseX_corrected = max(0, newPoseX)
      newPoseX_corrected = min(self.sizeX-1, newPoseX_corrected)
      newPoseY_corrected = max(0, newPoseY)
      newPoseY_corrected = min(self.sizeY-1, newPoseY_corrected)
      newState_corrected = self.getLinearIndex(newPoseX_corrected, newPoseY_corrected)
      if(newState_corrected in self.terminalStates):
        # Terminal state
        newState = newState_corrected
        reward = self.finishReward
        done = True
        self.agentState = newState
        return newState, reward, done
      else:
        # Out of grid
        newState = self.reset()
        reward = self.outOfGridReward
        done = False
        self.agentState = newState
        return newState, reward, done
    else:
      # Inside grid
      newState = self.getLinearIndex(newPoseX, newPoseY)
      if(newState in self.terminalStates):
        # Terminal state
        reward = self.finishReward
        done = True
        self.agentState = newState
        return newState, reward, done
      elif(newState in self.impassableStates):
        # You shall not pass
        newState = self.reset()
        reward = self.crashReward
        done = False
        self.agentState = newState
        return newState, reward, done
      else:
        # Still in the track
        reward = self.defaultReward
        done = False
        self.agentState = newState
        return newState, reward, done

  def reset(self):
    if len(self.startStates)>0:
      self.agentState = np.random.choice(self.startStates)
    else:
      agentPos = np.array([np.random.randint(self.sizeX), np.random.randint(self.sizeY)])
      self.agentState = self.getLinearIndex(agentPos[0], agentPos[1])
    return self.agentState
    
  def getAgentState(self):
    return self.agentState
  
  def getAgentPos(self):
    return self.getCartesianIndex(self.agentState)
    
  def getLinearIndex(self, x, y, size=None):
    if(size is None):
      return y*self.sizeX + x
    else:
      return y*size + x
    
  def getCartesianIndex(self, idx, size=None):
    if(size is None):
      x = idx%self.sizeX
      y = idx//self.sizeX
    else:
      x = idx%size
      y = idx//size
    
    return np.array([x,y])

  def getAvailableActions(self, state=None):
    return np.array(range(self.nActions))
  
  def drawEnv(self, agent=None):
    print_str = ""
    for y in range(self.sizeY):
      for x in range(self.sizeX):
        idx_state = self.getLinearIndex(x, y)
        if(idx_state in self.startStates and agent is None):
          print_str += " S "
        elif(idx_state in self.terminalStates):
          print_str += " F "
        elif(idx_state in self.impassableStates):
          print_str += " X "
        else:
          if(agent is None):
            print_str += "   "
          else:
            action = agent.getGreedyAction(idx_state)
            action_str = self.actionMapping[action][1]
            action_str_cropped = action_str[0:min(len(action_str)-1, self.VIS_NCHAR_PER_CELL)]
            print_str += " " * max(0,(self.VIS_NCHAR_PER_CELL-len(action_str_cropped)-1)) + action_str # TODO: center
      print_str += "\n"
    return print_str

  def printEnv(self, agent=None):
    print("SizeX, SizeY: ", self.sizeX, " ", self.sizeY)
    print()
    print_str = self.drawEnv(agent)
    print(print_str)

  def render(self, trajectory=None):
    print_str = self.drawEnv()
    print_str_list = list(print_str)
    if(trajectory is None):
      agentPose = self.getCartesianIndex(self.agentState)
      print_str_list[self.getLinearIndex(agentPose[0]*3, agentPose[1], (self.sizeX)*3+1)] = "*"
    else:
      for idx_state in range(len(trajectory)):
        agentPose = self.getCartesianIndex(trajectory[idx_state])
        print_str_list[self.getLinearIndex(agentPose[0]*3, agentPose[1], (self.sizeX)*3+1)] = "*"
    print("".join(print_str_list)) 

class DeterministicGridWorld(GridWorld):

  def __init__(self, sizeX, sizeY, startStates=[], terminalStates=[], impassableStates=[], 
    specialStateTransitions={}, defaultReward=-1.0, outOfGridReward=None, specialRewards={}):  
    super().__init__(sizeX, sizeY, startStates=startStates, terminalStates=terminalStates, impassableStates=impassableStates,
      specialStateTransitions=specialStateTransitions, defaultReward=defaultReward, crashReward=None, finishReward=None, 
      outOfGridReward=outOfGridReward, specialRewards=specialRewards)
    self.generateModels()

  def setImpassableStates(self, impassableStates):
    self.impassableStates = [self.getLinearIndex(s[0], s[1]) for s in impassableStates]
  
  def generateModels(self):
    self.stateTransitionProbs = np.zeros([self.nStates, self.nActions, self.nStates], dtype=np.float)
    self.rewardFunction = np.zeros([self.nStates, self.nActions], dtype=np.float) + self.defaultReward 
    for idx_state in range(self.nStates):
      s1 = self.getCartesianIndex(idx_state)
      if idx_state in self.terminalStates:
        self.stateTransitionProbs[idx_state,:,:] = 0.0
      elif idx_state in self.specialStateTransitions.keys():
        self.stateTransitionProbs[idx_state,:,self.specialStateTransitions[idx_state]] = 1.0
      else:
        for idx_action in range(self.nActions):
          a = self.actionMapping[idx_action][0]
          s2 = s1 + a
          if(s2[0]<0 or s2[0]>=self.sizeX or s2[1]<0 or s2[1]>=self.sizeY):
            # Out of grid
            self.stateTransitionProbs[idx_state, idx_action, idx_state] = 1.0
            self.rewardFunction[idx_state,idx_action] = self.outOfGridReward
          elif(self.getLinearIndex(s2[0],s2[1]) in self.impassableStates):
            self.stateTransitionProbs[idx_state, idx_action, idx_state] = 1.0
            self.rewardFunction[idx_state,idx_action] = self.outOfGridReward
          else:
            next_state = self.getLinearIndex(s2[0], s2[1])
            self.stateTransitionProbs[idx_state, idx_action, next_state] = 1.0
            self.rewardFunction[idx_state,idx_action] = self.defaultReward
    for state, action in self.specialRewards.keys():
      self.rewardFunction[state,action] = self.specialRewards[state,action]
      
  def step(self, action):
    state = self.getAgentState()
    newState = np.argmax(self.stateTransitionProbs[state, action,:]) # Deterministic 
    reward = self.rewardFunction[state, action]
    self.agentState = newState
    done = True if newState in self.terminalStates else False
    return newState, reward, done
  
  def computeExpectedValue(self, idx_state, idx_action, valueTable, gamma):
    reward = self.rewardFunction[idx_state, idx_action]
    sum_expect_nextstates = 0.0
    for idx_next_state in range(self.nStates):
      prob_next_state = self.stateTransitionProbs[idx_state, idx_action, idx_next_state]
      sum_expect_nextstates += prob_next_state * (reward + gamma * valueTable[idx_next_state])
    return sum_expect_nextstates
    
class StochasticGridWorld(GridWorld):

  def __init__(self, sizeX, sizeY, startStates=[], terminalStates=[], impassableStates=[],
    defaultReward=-1.0, crashReward=None, finishReward=None, outOfGridReward=None, specialRewards={},
    actionMapping=None, actionNoiseParams={}):
    super().__init__(sizeX, sizeY, startStates=startStates, terminalStates=terminalStates, impassableStates=impassableStates,
      defaultReward=defaultReward, crashReward=None, finishReward=None, outOfGridReward=outOfGridReward, specialRewards=specialRewards)
    self.actionNoiseParams = {"mean":[np.zeros(self.nStates), np.zeros(self.nStates)], "sigma":[np.zeros(self.nStates), np.zeros(self.nStates)]}
    for k, v in actionNoiseParams.items():
      idx_state = self.getLinearIndex(k[0],k[1])
      self.actionNoiseParams["mean"][0][idx_state] = v[0]
      self.actionNoiseParams["mean"][1][idx_state] = v[1]
      self.actionNoiseParams["sigma"][0][idx_state] = v[2]
      self.actionNoiseParams["sigma"][1][idx_state] = v[3]

  def step(self, action):
    agentPose = self.getCartesianIndex(self.agentState)
    actionOnGrid = self.actionMapping[action][0]
    actionNoiseX = np.random.normal(self.actionNoiseParams["mean"][0][self.agentState], self.actionNoiseParams["sigma"][0][self.agentState])
    actionNoiseY = np.random.normal(self.actionNoiseParams["mean"][1][self.agentState], self.actionNoiseParams["sigma"][1][self.agentState])   
    actionWithNoise = actionOnGrid + np.array([actionNoiseX, actionNoiseY], dtype=np.int)
    newPoseX = agentPose[0] + actionWithNoise[0]
    newPoseY = agentPose[1] + actionWithNoise[1]
    if(newPoseX<0 or newPoseX>=self.sizeX or newPoseY<0 or newPoseY>=self.sizeY):
      # Out of grid
      newPoseX_corrected = max(0, newPoseX)
      newPoseX_corrected = min(self.sizeX-1, newPoseX_corrected)
      newPoseY_corrected = max(0, newPoseY)
      newPoseY_corrected = min(self.sizeY-1, newPoseY_corrected)
      newState_corrected = self.getLinearIndex(newPoseX_corrected, newPoseY_corrected)
      if(newState_corrected in self.terminalStates):
        newState = newState_corrected
        reward = self.finishReward
        done = True
        self.agentState = newState
        return newState, reward, done
      else:
        # Out of grid
        newState = newState_corrected
        reward = self.outOfGridReward
        done = False
        self.agentState = newState
        return newState, reward, done
    else:
      # Inside grid
      newState = self.getLinearIndex(newPoseX, newPoseY)
      if(newState in self.terminalStates):
        if ((self.agentState,action),newState) in self.specialRewards.keys():
          reward = self.specialRewards[((self.agentState,action),newState)]
        else:
          reward = self.finishReward
        done = True
        self.agentState = newState
        return newState, reward, done
      else:
        # Still in the track
        if ((self.agentState,action),newState) in self.specialRewards.keys():
          reward = self.specialRewards[((self.agentState,action),newState)]
        else:
          reward = self.defaultReward
        done = False
        self.agentState = newState
        return newState, reward, done
    
class RaceTrack(GridWorld):

  VELOCITY_MIN = 0
  VELOCITY_MAX = 5
  
  def __init__(self, sizeX, sizeY, startStates=[], terminalStates=[], impassableStates=[],
    defaultReward=-1.0, crashReward=None, finishReward=None, outOfGridReward=None,
    specialRewards={}, p_actionFail=0.1):   
    # States
    self.sizeX = sizeX
    self.sizeY = sizeY
    self.outOfTrackState = self.sizeX*self.sizeY
    self.nStates = self.sizeX*self.sizeY+1
    self.startStates = []
    self.startStates = [self.getLinearIndex(s[0], s[1]) for s in startStates]
    self.terminalStates = [self.getLinearIndex(s[0], s[1]) for s in terminalStates]
    self.impassableStates = [self.getLinearIndex(s[0], s[1])for s in impassableStates]  
    # Actions
    self.actionMapping = {0:(np.array([+1,+1]), "HAVA"), 1:(np.array([0,+1]), "H0VA"), 2:(np.array([-1,+1]), "HDVA"),\
      3:(np.array([+1,0]), "HAV0"), 4:(np.array([0,0]), "H0V0"), 5:(np.array([-1,0]), "HDV0"),\
      6:(np.array([+1,-1]), "HAVD"), 7:(np.array([0,-1]), "H0VD"), 8:(np.array([-1,-1]), "HDVD")}   
    self.nActions = len(self.actionMapping)
    self.velocities = range(self.VELOCITY_MIN, self.VELOCITY_MAX+1)
    self.nVelocities = len(self.velocities)
    self.dimVelocities = self.nVelocities * self.nVelocities
    self.p_actionFail = p_actionFail
    # Rewards
    self.defaultReward = defaultReward
    self.finishReward = finishReward if finishReward is not None else defaultReward
    self.outOfGridReward = outOfGridReward if outOfGridReward is not None else defaultReward
    self.crashReward = crashReward if crashReward is not None else defaultReward
    self.specialRewards = {(self.getLinearIndex(k[0][0],k[0][1]),k[1]):v for k, v in specialRewards.items()}        
    # Agent state
    self.agentVelocity = np.array([0, 0], dtype=np.int)
    self.agentState = np.random.choice(self.startStates)
    
  def step(self, action):
    agentPose = self.getCartesianIndex(self.agentState)
    if(np.random.random()<self.p_actionFail):
      dVelocity = np.array([0, 0], dtype=np.int)
    else:
      dVelocity = self.actionMapping[action][0]
    self.agentVelocity += dVelocity
    newPoseX = agentPose[0] + self.agentVelocity[0]
    newPoseY = agentPose[1] - self.agentVelocity[1]
    if(newPoseX<0 or newPoseX>=self.sizeX or newPoseY<0 or newPoseY>=self.sizeY):
      # Out of grid
      newPoseX_corrected = max(0, newPoseX)
      newPoseX_corrected = min(self.sizeX-1, newPoseX_corrected)
      newPoseY_corrected = max(0, newPoseY)
      newPoseY_corrected = min(self.sizeY-1, newPoseY_corrected)
      newState_corrected = self.getLinearIndex(newPoseX_corrected, newPoseY_corrected)
      if(newState_corrected in self.terminalStates):
        # Crossed the finish line
        newState = newState_corrected
        reward = self.finishReward
        done = True
        self.agentState = newState
        return newState, reward, done
      else:
        # Out of grid
        newState = self.reset()
        if(self.crashReward is not None):
          reward = self.crashReward
        else:
          reward = self.defaultReward
        done = False
        self.agentState = newState
        return newState, reward, done
    else:
      # Inside grid
      newState = self.getLinearIndex(newPoseX, newPoseY)
      if(newState in self.terminalStates):
        # Crossed the finish line
        reward = self.finishReward
        done = True
        self.agentState = newState
        return newState, reward, done
      elif(newState in self.impassableStates):
        # Out of track
        newState = self.reset()
        if(self.crashReward is not None):
          reward = self.crashReward
        else:
          reward = self.defaultReward
        done = False
        self.agentState = newState
        return newState, reward, done
      else:
        # Still in the track
        reward = self.defaultReward
        done = False
        self.agentState = newState
        return newState, reward, done
    
  def getAllowedActionsMask(self, state=None):
    if(state is None):
      state = self.agentState
    maskActionsAllowed = np.ones(self.nActions)
    agentPose = self.getCartesianIndex(state)
    for idx_action in range(self.nActions):
      dVelocity = self.actionMapping[idx_action][0]
      newVelocity = self.agentVelocity + dVelocity
      if(newVelocity[0]<self.VELOCITY_MIN or newVelocity[0]>self.VELOCITY_MAX or \
         newVelocity[1]<self.VELOCITY_MIN or newVelocity[1]>self.VELOCITY_MAX):
        maskActionsAllowed[idx_action] = 0
      if(newVelocity[0]==0 and newVelocity[1]==0):
        maskActionsAllowed[idx_action] = 0
    return maskActionsAllowed
    
  def getAvailableActions(self, state=None):
    return np.nonzero(self.getAllowedActionsMask(state))[0]
    
  def reset(self):
    self.agentVelocity = np.array([0, 0], dtype=np.int)
    self.agentState = np.random.choice(self.startStates)
    return self.agentState
    
  def printEnv(self, agent=None):
    print("SizeX, SizeY: ", self.sizeX, " ", self.sizeY)
    print()
    print_str_X, print_str_Y = self.drawEnv(agent)
    print(print_str_X)
    if(agent is not None):
      print()
      print(print_str_Y)

  def drawEnv(self, agent=None):
    print_str_X = ""
    print_str_Y = ""
    for y in range(self.sizeY):
      for x in range(self.sizeX):
        idx_lin = self.getLinearIndex(x, y)
        if(idx_lin in self.startStates and agent is None):
          print_str_X += " S "
          print_str_Y += " S "
        elif(idx_lin in self.terminalStates):
          print_str_X += " F "
          print_str_Y += " F "
        elif(idx_lin in self.impassableStates):
          print_str_X += " X "
          print_str_Y += " X "
        else:
          if(agent is None):
            print_str_X += "   "
            print_str_Y += "   "
          else:
            action = agent.getGreedyAction(idx_lin)
            vx = self.actionMapping[action][0][0]
            vy = self.actionMapping[action][0][1]
            if(vx<0):
              vx_str = str(vx) + " "
            else:
              vx_str = "+" + str(vx) + " "
            if(vy<0):
              vy_str = str(vy) + " "
            else:
              vy_str = "+" + str(vy) + " "
            print_str_X += vx_str
            print_str_Y += vy_str          
      print_str_X += "\n"
      print_str_Y += "\n"
    return print_str_X, print_str_Y
    
  def render(self, agentHistory=None):
    print_str, _ = self.drawEnv()
    print_str_list = list(print_str)
    if(agentHistory is None):
      agentPose = self.getCartesianIndex(self.agentState)
      print_str_list[self.getLinearIndex(agentPose[0]*3, agentPose[1], (self.sizeX)*3+1)] = "*"
    else:
      for idx_state in range(len(agentHistory)):
        agentPose = self.getCartesianIndex(agentHistory[idx_state])
        print_str_list[self.getLinearIndex(agentPose[0]*3, agentPose[1], (self.sizeX)*3+1)] = "*"
    print("".join(print_str_list))
  