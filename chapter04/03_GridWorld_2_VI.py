'''
02_GridWorld_2_PI.py : replication of Figure 4.1 using Value Iteration

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.DynamicProgramming import ValueIteration

if __name__=="__main__":

  # Environment
  sizeX = 4
  sizeY = 4
  defaultReward = -1.0
  terminalStates= [(0,0), (3,3)]

  # Agent
  gamma = 0.9
  thresh_convergence = 0.01

  env = DeterministicGridWorld(sizeX, sizeY, defaultReward=defaultReward, terminalStates=terminalStates)
  agent = ValueIteration(env.nStates, env.nActions, gamma, thresh_convergence, env.computeExpectedValue)
  
  env.printEnv()
  
  deltaMax, isConverged, isPolicyStable = agent.update()
  
  print("Delta: ", deltaMax, " isConverged: ", isConverged, " isPolicyStable: ", isPolicyStable)
  
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
    
  env.printEnv(agent)
