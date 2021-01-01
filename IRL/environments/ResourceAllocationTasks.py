'''
ResourceAllocationTasks.py : basic resource allocation tasks

Cem Karaoguz, 2020
MIT License
'''

import numpy as np

class JacksCarRental:
  
  N_CARS_MAX_REQUESTS = 7
  N_CARS_MAX_RETURNS = 7
  
  def __init__(self, nCarsMaxA=20, nCarsMaxB=20, nCarsMaxRelocate=5, lambdaCarRequestA=3,
    lambdaCarRequestB=4, lambdaCarReturnA=3, lambdaCarReturnB=2, rewardRequest=10.0,
    rewardRelocation=-2.0, nCarFreeRelocation=0, nCarsParkingLimit=100, reward_extraParkingFees=0.0):
    self.nCarsMaxA = nCarsMaxA
    self.nCarsMaxB = nCarsMaxB
    self.nCarsASize = nCarsMaxA+1
    self.nCarsBSize = nCarsMaxB+1    
    self.nCarsMaxRelocate = nCarsMaxRelocate
    self.lambdaCarRequestA = lambdaCarRequestA
    self.lambdaCarRequestB = lambdaCarRequestB
    self.lambdaCarReturnA = lambdaCarReturnA
    self.lambdaCarReturnB = lambdaCarReturnB
    self.rewardRequest = rewardRequest
    self.rewardRelocation = rewardRelocation
    self.nCarFreeRelocation = nCarFreeRelocation
    self.nCarsParkingLimit = nCarsParkingLimit
    self.reward_extraParkingFees = reward_extraParkingFees
    self.nStates = self.nCarsASize * self.nCarsBSize
    self.actionMapping = {}
    for i in range(self.nCarsMaxRelocate*2+1):
      self.actionMapping[i] = (i-self.nCarsMaxRelocate, str(i-self.nCarsMaxRelocate))
    self.nActions = len(self.actionMapping)
    # State transitions and reward function
    self.probRequestsA = self.prob_poisson(np.array(range(self.nCarsASize), dtype=np.float), self.lambdaCarRequestA)
    self.probRequestsB = self.prob_poisson(np.array(range(self.nCarsBSize), dtype=np.float), self.lambdaCarRequestB)
    self.probReturnsA = self.prob_poisson(np.array(range(self.nCarsASize), dtype=np.float), self.lambdaCarReturnA)
    self.probReturnsB = self.prob_poisson(np.array(range(self.nCarsBSize), dtype=np.float), self.lambdaCarReturnB)
    self.actionsAllowed = np.zeros([self.nStates, self.nActions], dtype=np.int)
    for idx_state in range(self.nStates):
      for idx_action in range(self.nActions):
        nCarsA, nCarsB = self.getCartesianIndex(idx_state)
        action = self.actionMapping[idx_action][0]
        nCarsA_next = nCarsA + action
        nCarsB_next = nCarsB - action
        if(nCarsA_next>=0 and nCarsB_next>=0):
          self.actionsAllowed[idx_state, idx_action] = 1
    self.nCarsA = np.random.randint(self.nCarsASize)
    self.nCarsB = np.random.randint(self.nCarsBSize)

  def step(self, action):
    # Night
    state = self.getLinearIndex(self.nCarsA, self.nCarsB)
    carsToMoveToA = self.actionMapping[action][0]
    if(carsToMoveToA>0 and carsToMoveToA>self.nCarsB):
      carsToMoveToA = self.nCarsB
      action = self.findActionIndex(carsToMoveToA)
    if(carsToMoveToA<0 and abs(carsToMoveToA)>self.nCarsA):
      carsToMoveToA = self.nCarsA
      action = self.findActionIndex(carsToMoveToA)
    self.nCarsA = self.nCarsA + carsToMoveToA
    self.nCarsB = self.nCarsB - carsToMoveToA
    self.nCarsA = min(self.nCarsA, self.nCarsMaxA)
    self.nCarsB = min(self.nCarsB, self.nCarsMaxB)
    # Day
    nCarsRequestedA = min(np.random.poisson(self.lambdaCarRequestA), self.nCarsA)
    nCarsRequestedB = min(np.random.poisson(self.lambdaCarRequestB), self.nCarsB)
    nCarsReturnedA = np.random.poisson(self.lambdaCarReturnA)
    nCarsReturnedB = np.random.poisson(self.lambdaCarReturnB)
    self.nCarsA = self.nCarsA - nCarsRequestedA + nCarsReturnedA
    self.nCarsB = self.nCarsB - nCarsRequestedB + nCarsReturnedB
    self.nCarsA = min(self.nCarsA, self.nCarsMaxA)
    self.nCarsB = min(self.nCarsB, self.nCarsMaxB)
    newState = self.getLinearIndex(self.nCarsA, self.nCarsB)
    reward = (nCarsRequestedA + nCarsRequestedB) * self.rewardRequest + abs(carsToMoveToA) * self.rewardRelocation
    done = True
    return newState, reward, done, action
  
  def reset(self):
    self.nCarsA = np.random.randint(self.nCarsASize)
    self.nCarsB = np.random.randint(self.nCarsBSize)
    return self.getLinearIndex(self.nCarsA, self.nCarsB)
    
  def getAvailableActions(self, state=None):
    if(state is None):
      state = self.getLinearIndex(self.nCarsA, self.nCarsB)
    return np.nonzero(self.actionsAllowed[state,:])[0]

  def computeExpectedValue(self, idx_state, idx_action, valueTable, gamma):
    returnVal = 0.0
    nCarsA, nCarsB = self.getCartesianIndex(idx_state)
    action = self.actionMapping[idx_action][0]
    nCarsA_next = nCarsA + action
    nCarsB_next = nCarsB - action
    nCarsA_next = min(nCarsA_next, self.nCarsMaxA)
    nCarsB_next = min(nCarsB_next, self.nCarsMaxB)
    
    # Relocation reward
    if(action<0):
      netCarsMoved = max(np.abs(action) - self.nCarFreeRelocation, 0) # Excercise 4.7
    else:
      netCarsMoved = np.abs(action)
    reward_relocation = netCarsMoved * self.rewardRelocation
    
    # Excercise 4.7
    reward_extraParkingFees = 0.0
    if(nCarsA_next>self.nCarsParkingLimit):
      reward_extraParkingFees = reward_extraParkingFees + self.reward_extraParkingFees
    if(nCarsB_next>self.nCarsParkingLimit):
      reward_extraParkingFees = reward_extraParkingFees + self.reward_extraParkingFees
    returnVal = returnVal + reward_relocation + reward_extraParkingFees
    # Rentals
    for nRentalsA in range(self.N_CARS_MAX_REQUESTS):
      for nRentalsB in range(self.N_CARS_MAX_REQUESTS):
        nRentalsA_total = min(nRentalsA, nCarsA_next)
        nRentalsB_total = min(nRentalsB, nCarsB_next)
        reward_request = (nRentalsA_total+nRentalsB_total)*self.rewardRequest
        # Returns
        for nReturnsA in range(self.N_CARS_MAX_RETURNS):
          for nReturnsB in range(self.N_CARS_MAX_RETURNS):
            nCarsA_final = min(nCarsA_next - nRentalsA_total + nReturnsA, self.nCarsMaxA)
            nCarsB_final = min(nCarsB_next - nRentalsB_total + nReturnsB, self.nCarsMaxB)
            prob_st = self.probRequestsA[nRentalsA] * self.probRequestsB[nRentalsB] * self.probReturnsA[nReturnsA] * self.probReturnsB[nReturnsB]
            returnVal = returnVal + prob_st * ( reward_request + gamma * valueTable[self.getLinearIndex(nCarsA_final, nCarsB_final)])
    return returnVal
  
  def getLinearIndex(self, nCarsA, nCarsB):
    return nCarsB*self.nCarsASize + nCarsA

  def getCartesianIndex(self, idx):
    nCarsA = idx%self.nCarsASize
    nCarsB = idx//self.nCarsASize
    return nCarsA, nCarsB

  def findActionIndex(self, action):
    actionIndex = None
    for i in range(self.nActions):
      if(self.actionMapping[i][0]==action):
        actionIndex=i
        break
    return actionIndex
  
  def prob_poisson(self, n, lmd):
    if(not isinstance(n, int)):
      factorials = []
      for i in n:
        factorials.append(np.math.factorial(i))
      return np.power(lmd, n) * np.exp(-lmd) / np.array(factorials, dtype=np.float)
    else:
      return  np.power(lmd, n) * np.exp(-lmd) / np.math.factorial(n)
    
  def printEnv(self, agent=None):
    print("nCarsMaxA, nCarsMaxB: ", self.nCarsMaxA, " ", self.nCarsMaxB)
    print("nCarsMaxRelocate:", self.nCarsMaxRelocate)
    print("lambdaCarRequestA, lambdaCarRequestB: ", self.lambdaCarRequestA, " ", self.lambdaCarRequestB)
    print("lambdaCarReturnA, lambdaCarReturnB: ", self.lambdaCarReturnA, " ", self.lambdaCarReturnB)
    print("nStates, nActions: ", self.nStates, " ", self.nActions)
    print("Action mapping:\n", self.actionMapping)
    print()
    
    if(agent is not None):
      printMatrix_pol = np.zeros([self.nCarsMaxA+1, self.nCarsMaxB+1], dtype=np.int)
      printMatrix_val = np.zeros([self.nCarsMaxA+1, self.nCarsMaxB+1], dtype=np.int)
      for y in range(self.nCarsMaxB+1):
        for x in range(self.nCarsMaxA+1):
          i = self.getLinearIndex(x,y)
          action = agent.selectAction(i)
          printMatrix_pol[self.nCarsMaxA-x,y] = self.actionMapping[action][0]
          printMatrix_val[self.nCarsMaxA-x,y] = agent.getValue(i)
      print(printMatrix_val)
      print()
      print(printMatrix_pol)
      
class AccessControlTask:
  
  ACTION_IDX_ACCEPT = 0
  ACTION_IDX_REJECT = 1
  
  def __init__(self, nServers=10, customerPriorities=[1,2,4,8], pServerFree=0.06, rejectionReward=0.0, doLinearStateIndexing=False):
    
    self.nServers = nServers
    self.pServerFree = pServerFree
    self.rejectionReward = rejectionReward
    if doLinearStateIndexing==True:
      self.encodeState = self.getLinearIndex
    else:
      self.encodeState = lambda x: x
    self.stateRewardMapping = {i:customerPriorities[i] for i in range(len(customerPriorities))}
    self.dimStates = [len(self.stateRewardMapping), self.nServers]
    self.nStates = np.prod(self.dimStates)
    self.actionMapping = {self.ACTION_IDX_ACCEPT:(0, "Accept"), self.ACTION_IDX_REJECT:(1, "Reject")}
    self.nActions = len(self.actionMapping)
    self.nFreeServers = None
    self.queueHead = None
    self.reset()
  
  def step(self, action):
    # Continuous task
    done = False
    # Update server availability
    nOccupiedServers = self.nServers - self.nFreeServers
    nServersBecameFree = np.sum(np.random.rand(nOccupiedServers)<self.pServerFree)
    self.nFreeServers += nServersBecameFree
    assert self.nFreeServers<=self.nServers
    # Handle queue head
    if self.nFreeServers==0:
      # No free servers, reject customer
      reward = self.rejectionReward
    else:
      if action==self.ACTION_IDX_ACCEPT:
        self.nFreeServers-=1
        reward = self.stateRewardMapping[self.queueHead]
      elif action==self.ACTION_IDX_REJECT:
        reward = self.rejectionReward
      else:
        assert False, "Unrecognized action!"
    # Get new customer
    self.queueHead = self.getCustomer()
    return self.encodeState([self.queueHead, self.nFreeServers]), reward, done
  
  def getCustomer(self):
    return np.random.choice(list(self.stateRewardMapping.keys()))
    
  def reset(self):
    self.queueHead = self.getCustomer()
    self.nFreeServers = self.nServers
    return self.encodeState([self.queueHead, self.nFreeServers])
    
  def getLinearIndex(self, state):
    [queueHead, nFreeServers] = state
    return queueHead*self.nServers + nFreeServers

  def getCartesianIndex(self, idx):
    nFreeServers = idx%self.nServers
    queueHead = idx//self.nServers
    return [queueHead, nFreeServers]