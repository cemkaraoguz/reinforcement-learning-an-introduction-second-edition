'''
04_JCR_PI.py : replication of Figure 4.2

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.ResourceAllocationTasks import JacksCarRental
from IRL.agents.DynamicProgramming import PolicyIteration

if __name__=="__main__":

  nEpisodes = 1000
  doAddComplexity = False  # set True for Excercise 4.7
  
  # Environment
  nCarsMaxA = 20
  nCarsMaxB = 20
  nCarsMaxRelocate = 5
  lambdaCarRequestA = 3
  lambdaCarRequestB = 4
  lambdaCarReturnA = 3
  lambdaCarReturnB = 2
  rewardRequest = 10.0
  rewardRelocation = -2.0

  if(doAddComplexity):
    # Excercise 4.7
    nCarFreeRelocation = 1
    nCarsParkingLimit = 10
    reward_extraParkingFees = -4.0
  else:
    nCarFreeRelocation = 0
    nCarsParkingLimit = 50
    reward_extraParkingFees = 0.0
    
  # Agent
  gamma = 1.0
  thresh_convergence = 1e-30

  env = JacksCarRental(nCarsMaxA=nCarsMaxA, nCarsMaxB=nCarsMaxB, nCarsMaxRelocate=nCarsMaxRelocate,
    lambdaCarRequestA=lambdaCarRequestA, lambdaCarRequestB=lambdaCarRequestB,
    lambdaCarReturnA=lambdaCarReturnA, lambdaCarReturnB=lambdaCarReturnB,
    rewardRequest=rewardRequest, rewardRelocation=rewardRelocation, nCarFreeRelocation=nCarFreeRelocation,
    nCarsParkingLimit=nCarsParkingLimit, reward_extraParkingFees=reward_extraParkingFees)
  agent = PolicyIteration(env.nStates, env.nActions, gamma, thresh_convergence,
    env.computeExpectedValue, env.actionsAllowed, iterationsMax=1000)
  
  env.printEnv()
  
  for e in range(nEpisodes):
      
    deltaMax, isConverged, isPolicyStable = agent.update()
    
    print("Episode : ", e, " Delta: ", deltaMax, " isConverged: ", isConverged, " isPolicyStable: ", isPolicyStable)
        
    env.printEnv(agent)
    
    if(isPolicyStable):
      print("Stable policy achieved!")
      break