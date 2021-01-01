'''
01_Blackjack_MCC.py : replication of Figure 5.2

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from IRL.environments.Gambler import Blackjack
from IRL.agents.MonteCarlo import MonteCarloControl

if __name__=="__main__":

  nEpisodes = 500000
  
  # Agent
  gamma = 1.0
  doUseAllVisits = False
  
  # Policy
  policyUpdateMethod = "esoft"
  epsilon = 0.1

  env = Blackjack()
  agent = MonteCarloControl(env.nStates, env.nActions, gamma, doUseAllVisits=doUseAllVisits, 
    policyUpdateMethod=policyUpdateMethod, epsilon=epsilon)
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
    
      action = agent.selectAction(state)

      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done
      
      new_state, reward, done = env.step(action)

      #print("Episode : ", e, " State : ", state, " Action : ", env.actionMapping[action][1], " Reward : ", reward, " Next state : ", new_state)
      
      xp = {}
      xp['reward'] = reward
      xp['state'] = new_state
      xp['done'] = done
      experiences.append(xp)

      state = new_state
      
    agent.update(experiences)
  
  value_usableace = np.zeros([env.nStatesPlayerSum, env.nStatesDealerShowing])
  value_nousableace = np.zeros([env.nStatesPlayerSum, env.nStatesDealerShowing])
  print_str_usableace = ""
  print_str_nousableace = ""
  for i in range(env.nStatesPlayerSum-1, -1, -1):
    for j in range(env.nStatesDealerShowing):
      idx_usableace = env.getLinearIndex(env.minPlayerSum+i, env.minDealerShowing+j, env.USABLE_ACE_YES)
      idx_nousableace = env.getLinearIndex(env.minPlayerSum+i, env.minDealerShowing+j, env.USABLE_ACE_NO)
      action_usableace = agent.getGreedyAction(idx_usableace)
      action_nousableace = agent.getGreedyAction(idx_nousableace)
      value_usableace[j,i] = agent.getValue(idx_usableace)
      value_nousableace[j,i] = agent.getValue(idx_nousableace)
      print_str_usableace += str(env.actionMapping[action_usableace][1]) + "\t"
      print_str_nousableace += str(env.actionMapping[action_nousableace][1]) + "\t"
    print_str_usableace += "\n"
    print_str_nousableace += "\n"

  print("Value function (usable Ace)")
  print(value_usableace)
  print() 
  print("Value function (No usable Ace)")
  print(value_nousableace)
  
  print("Policy (usable Ace)")
  print(print_str_usableace)
  print() 
  print("Policy (No usable Ace)")
  print(print_str_nousableace)
  
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
  