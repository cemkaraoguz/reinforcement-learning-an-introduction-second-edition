'''
02_Blackjack_MCP_OffP.py : Application of an off-policy Monte-Carlo method 
for Blackjack prediction problem (Figure 5.1)

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from IRL.environments.Gambler import Blackjack
from IRL.agents.MonteCarlo import MonteCarloOffPolicyPrediction
from IRL.utils.Policies import StochasticPolicy

if __name__=="__main__":

  nEpisodes = 500000

  # Agent
  gamma = 1.0
  
  env = Blackjack()
  agent = MonteCarloOffPolicyPrediction(env.nStates, env.nActions, gamma)
  policy_behaviour = StochasticPolicy(env.nStates, env.nActions)
  for i in range(env.nStatesPlayerSum-1, -1, -1):
    for j in range(env.nStatesDealerShowing):
      for k in [env.USABLE_ACE_YES, env.USABLE_ACE_NO]:
        idx_state = env.getLinearIndex(env.minPlayerSum+i, env.minDealerShowing+j, k)
        if(env.minPlayerSum+i<20):
          actionProb = np.zeros(env.nActions)
          actionProb[env.ACTION_HIT] = 1.0
          agent.policy.update(idx_state, actionProb)
        else:
          actionProb = np.zeros(env.nActions)
          actionProb[env.ACTION_STICK] = 1.0
          agent.policy.update(idx_state, actionProb)
  
  #env.printEnv()
  
  for e in range(nEpisodes):
    
    if(e%10000==0):
      print("Episode : ", e)

    experiences = [{}]
    state = env.reset()
    done = False
    while not done:
    
      action = policy_behaviour.sampleAction(state)
      
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
      
    agent.evaluate(experiences, policy_behaviour)

  value_usableace = np.zeros([env.nStatesPlayerSum, env.nStatesDealerShowing])
  value_nousableace = np.zeros([env.nStatesPlayerSum, env.nStatesDealerShowing])
  for i in range(env.nStatesPlayerSum-1, -1, -1):
    for j in range(env.nStatesDealerShowing):
      idx_usableace = env.getLinearIndex(env.minPlayerSum+i, env.minDealerShowing+j, env.USABLE_ACE_YES)
      idx_nousableace = env.getLinearIndex(env.minPlayerSum+i, env.minDealerShowing+j, env.USABLE_ACE_NO)
      value_usableace[j,i] = agent.getValue(idx_usableace)
      value_nousableace[j,i] = agent.getValue(idx_nousableace)
      
  X = np.arange(0, env.nStatesPlayerSum, 1)+12
  Y = np.arange(0, env.nStatesDealerShowing, 1)
  X, Y = np.meshgrid(X, Y)
  
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
  