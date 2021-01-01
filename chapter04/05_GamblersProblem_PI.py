'''
05_GamblersProblem_PI.py : replication of Figure 4.3 via Policy Iteration

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gambler import CoinFlipGame
from IRL.agents.DynamicProgramming import PolicyIteration

nSweeps = 50
visualizationSweeps = [1,5]

# Environment
maxCapital = 100
prob_heads = 0.4

# Agent
gamma = 1.0
thresh_convergence = 1e-30

if __name__=="__main__":

  env = CoinFlipGame(maxCapital, prob_heads)  
  agent = PolicyIteration(env.nStates, env.nActions, gamma, thresh_convergence,
    env.computeExpectedValue, env.actionsAllowed, iterationsMax=1000, doLogValueTables=True)
    
  #env.printEnv()
  valueEstimates = []
  for e in range(nSweeps):
      
    deltaMax, isConverged, isPolicyStable = agent.update()
    
    print("Sweep : ", e, " Delta: ", deltaMax, " isConverged: ", isConverged, " isPolicyStable: ", isPolicyStable)
    
  pl.figure()
  for e in visualizationSweeps:
    if e<len(agent.valueTables):
      printMatrix_pol = np.zeros(env.nStates-2, dtype=np.int)
      printMatrix_val = np.zeros(env.nStates-2, dtype=np.float)
      for i in range(env.nStates-2):
        action = agent.selectAction(i+1, env.getAvailableActions())
        printMatrix_val[i] = agent.valueTables[e-1][i]
        printMatrix_pol[i] = action
    pl.plot(printMatrix_val, label="Sweep "+str(e))
  pl.xlabel("Capital")
  pl.ylabel("Value estimates")
  pl.legend()

  printMatrix_pol = np.zeros(env.nStates-2, dtype=np.int)
  printMatrix_val = np.zeros(env.nStates-2, dtype=np.float)
  for i in range(env.nStates-2):
    action = agent.selectAction(i+1, env.getAvailableActions())
    printMatrix_val[i] = agent.valueTable[i]
    printMatrix_pol[i] = action
  print("Final value estimates and policy")
  print(printMatrix_val)
  print()
  print(printMatrix_pol)

  pl.figure()
  pl.plot(printMatrix_pol)
  pl.xlabel("Capital")
  pl.ylabel("Final policy (stake)")
  pl.show()
  