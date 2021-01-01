'''
00_GridWorld_2_PE.py : replication of Figure 4.1

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
  sizeY = 4
  defaultReward = -1.0
  terminalStates= [(0,0), (3,3)]

  # Agent
  gamma = 1.0
  thresh_convergence = 1e-30
  
  env = DeterministicGridWorld(sizeX, sizeY, defaultReward=defaultReward, terminalStates=terminalStates)
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
