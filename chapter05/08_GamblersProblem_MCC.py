'''
08_GamblersProblem_MCC.py : Application of a Monte Carlo solution to Gambler's Problem (control)

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gambler import CoinFlipGame
from IRL.agents.MonteCarlo import MonteCarloControl

if __name__=="__main__":

  nEpisodes = 100000

  # Environment
  maxCapital = 100
  prob_heads = 0.4

  # Agent
  gamma = 1.0
  epsilon = 0.1

  env = CoinFlipGame(maxCapital, prob_heads)
  agent = MonteCarloControl(env.nStates, env.nActions, gamma, policyUpdateMethod="esoft", epsilon=epsilon, doUseAllVisits=False)
  
  #env.printEnv()
  
  for e in range(nEpisodes):
  
    if(e%1000==0):
      print("Episode : ", e)
    
    experiences = [{}]
    state = env.reset()
    done = False
    while not done:
    
      action = agent.selectAction(state, env.getAvailableActions())
      
      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done
      
      new_state, reward, done = env.step(action)

      #print("Episode : ", e, " State : ", state, " Action : ", action, " Reward : ", reward, " Next state : ", new_state)
      
      xp = {}
      xp['reward'] = reward
      xp['state'] = new_state
      xp['done'] = done
      experiences.append(xp)
      
      state = new_state
    
    agent.update(experiences)

  valueTable = np.zeros(env.nStates)
  greedyPolicy = np.zeros(env.nStates, dtype=int)
  for idx_state in range(env.nStates):
    valueTable[idx_state] = agent.getValue(idx_state)
    greedyPolicy[idx_state] = agent.getGreedyAction(idx_state, env.getAvailableActions(idx_state))

  pl.figure()
  pl.plot(valueTable[1:-1])
  pl.xlabel("Capital")
  pl.ylabel("Value estimates")
  pl.figure()
  pl.plot(greedyPolicy[1:-1])
  pl.xlabel("Capital")
  pl.ylabel("Final policy (stake)")  
  pl.show()