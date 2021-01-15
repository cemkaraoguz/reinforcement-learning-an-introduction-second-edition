'''
05_GamblersProblem_PI.py : replication of Figure 4.3 via Value Iteration and
solution to exercise 4.9

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gambler import CoinFlipGame
from IRL.agents.DynamicProgramming import ValueIteration

if __name__=="__main__":

  # Environment
  maxCapital = 100
  # Figure 4.3 --> prob_heads = 0.4
  # Exercise 4.9 --> prob_heads = 0.25 and prob_heads = 0.55
  prob_heads = 0.25
  visualizationSweeps = [1, 2, 3, 32, 100, 1000, 3000]
  
  # Agent
  gamma = 1.0
  thresh_convergence = 1e-30

  env = CoinFlipGame(maxCapital, prob_heads)
  agent = ValueIteration(env.nStates, env.nActions, gamma, thresh_convergence,
    env.computeExpectedValue, env.actionsAllowed, iterationsMax=5000, doLogValueTables=True)
    
  #env.printEnv()
  
  deltaMax, isConverged, isPolicyStable = agent.update()
  
  print("Delta: ", deltaMax, " isConverged: ", isConverged, " isPolicyStable: ", isPolicyStable)
  
  if(isPolicyStable):
    print("Stable policy achieved!")
    
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
  pl.title("p="+str(prob_heads))
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
  pl.title("p="+str(prob_heads))
  pl.show()
