'''
01_GridWorld_2_extra_PE.py : solution to exercise 4.2

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.DynamicProgramming import PolicyEvaluation
from IRL.utils.Policies import StochasticPolicy

if __name__=="__main__":

  nEpisodes = 1000

  # Environment
  sizeX = 4
  sizeY = 5
  defaultReward = -1.0
  terminalStates= [(0,0), (3,3)]
  doUnblockAdditionalState = True   # Set true for the second part of exercise 4.2
  
  # Agent
  gamma = 0.9
  thresh_convergence = 1e-30

  env = DeterministicGridWorld(sizeX, sizeY, defaultReward=defaultReward, terminalStates=terminalStates)
  
  # First part of exercise 4.2: addition of new state
  env.stateTransitionProbs[12,:,16:]=0.0
  env.stateTransitionProbs[13,:,16:]=0.0
  env.stateTransitionProbs[14,:,16:]=0.0
  env.stateTransitionProbs[15,:,16:]=0.0

  env.stateTransitionProbs[16,:,:]=0.0
  env.stateTransitionProbs[18,:,:]=0.0
  env.stateTransitionProbs[19,:,:]=0.0

  # State 15 of the question is indexed as 17 
  env.stateTransitionProbs[17,:,:]=0.0
  env.stateTransitionProbs[17,0,13]=1.0 #N
  env.stateTransitionProbs[17,1,17]=1.0 #S
  env.stateTransitionProbs[17,2,14]=1.0 #E
  env.stateTransitionProbs[17,3,12]=1.0 #W
  
  # Second part of exercise 4.2: changed dynamics of state 13:
  if(doUnblockAdditionalState):
    env.stateTransitionProbs[13,1,17]=1.0
  
  policy = StochasticPolicy(env.nStates, env.nActions)
  agent = PolicyEvaluation(env.nStates, env.nActions, gamma, thresh_convergence, env.computeExpectedValue)
  
  env.printEnv()
    
  for e in range(nEpisodes):
    
    deltaMax, isConverged = agent.evaluate(policy)
    
    print("Episode : ", e, " Delta: ", deltaMax)
    
    printStr = ""
    for y in range(sizeY):
      for x in range(sizeX):
        i = env.getLinearIndex(x,y)
        printStr += "{:.2f}".format(agent.valueTable[i]) + "\t"
      printStr += "\n"
    print(printStr)
      
    if(isConverged):
      print("Convergence achieved!")
      break
