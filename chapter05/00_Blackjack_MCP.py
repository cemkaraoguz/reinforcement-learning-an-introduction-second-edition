'''
00_Blackjack_MCP.py : replication of Figure 5.1

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from IRL.environments.Gambler import Blackjack
from IRL.agents.MonteCarlo import MonteCarloPrediction
from IRL.utils.Policies import ActionValuePolicy

if __name__=="__main__":

  nEpisodes = 500000

  # Agent
  gamma = 1.0

  env = Blackjack()
  
  policy = ActionValuePolicy(env.nStates, env.nActions, actionSelectionMethod="greedy")
  for i in range(env.nStatesPlayerSum-1, -1, -1):
    for j in range(env.nStatesDealerShowing):
      for k in [env.USABLE_ACE_YES, env.USABLE_ACE_NO]:
        idx_state = env.getLinearIndex(env.minPlayerSum+i, env.minDealerShowing+j, k)
        if(env.minPlayerSum+i<20):
          actionProb = np.zeros(env.nActions)
          actionProb[env.ACTION_HIT] = 1.0
          policy.update(idx_state, actionProb)
        else:
          actionProb = np.zeros(env.nActions)
          actionProb[env.ACTION_STICK] = 1.0
          policy.update(idx_state, actionProb)
  
  agent = MonteCarloPrediction(env.nStates, gamma, doUseAllVisits=False)
  
  #env.printEnv()
    
  for e in range(nEpisodes):
    
    if(e%10000==0):
      print("Episode : ", e)

    experiences = [{}]
    state = env.reset()
    done = False
    while not done:
      action = policy.selectAction(state)
      
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
      
    agent.evaluate(experiences)
  
  value_usableace = np.zeros([env.nStatesPlayerSum, env.nStatesDealerShowing])
  value_nousableace = np.zeros([env.nStatesPlayerSum, env.nStatesDealerShowing])
  Y = np.arange(0, env.nStatesPlayerSum, 1)+12
  X = np.arange(0, env.nStatesDealerShowing, 1)
  X, Y = np.meshgrid(X, Y)
  for i in range(env.nStatesPlayerSum):
    for j in range(env.nStatesDealerShowing):
      value_usableace[i,j] = agent.getValue(env.getLinearIndex(env.minPlayerSum+i, env.minDealerShowing+j, env.USABLE_ACE_YES))
      value_nousableace[i,j] = agent.getValue(env.getLinearIndex(env.minPlayerSum+i, env.minDealerShowing+j, env.USABLE_ACE_NO))
    
  fig = pl.figure()
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, Y, value_usableace, linewidth=0, antialiased=False)
  ax.set_zlim(-1.01, 1.01)
  ax.set_xlabel("Player sum")
  ax.set_ylabel("Dealer showing")
  ax.set_title("Usable Ace")

  fig = pl.figure()
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, Y, value_nousableace, linewidth=0, antialiased=False)
  ax.set_zlim(-1.01, 1.01)
  ax.set_xlabel("Player sum")
  ax.set_ylabel("Dealer showing")
  ax.set_title("No usable Ace")
  
  pl.show()