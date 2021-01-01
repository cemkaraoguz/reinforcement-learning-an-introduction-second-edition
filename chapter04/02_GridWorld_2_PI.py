'''
02_GridWorld_2_PI.py : replication of Figure 4.1 using Policy Iteration

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.DynamicProgramming import PolicyIteration

if __name__=="__main__":

  nEpisodes = 10

  # Environment
  sizeX = 4
  sizeY = 4
  defaultReward = -1.0
  terminalStates= [(0,0), (3,3)]

  # Agent
  gamma = 0.9
  thresh_convergence = 1e-30

  env = DeterministicGridWorld(sizeX, sizeY, defaultReward=defaultReward, terminalStates=terminalStates)
  agent = PolicyIteration(env.nStates, env.nActions, gamma, thresh_convergence, env.computeExpectedValue)
  
  env.printEnv()
    
  for e in range(nEpisodes):
      
    deltaMax, isConverged, isPolicyStable = agent.update()
    
    print("Episode : ", e, " Delta: ", deltaMax, " isConverged: ", isConverged, " isPolicyStable: ", isPolicyStable)
    
    print()
    printStr = ""
    for y in range(sizeY):
      for x in range(sizeX):
        i = env.getLinearIndex(x,y)
        printStr = printStr + "{:.2f}".format(agent.valueTable[i]) + "\t"
      printStr += "\n"
    print(printStr)
    
    if(isPolicyStable):
      print("Stable policy achieved!")
      break
      
  env.printEnv(agent)
