'''
00_SimpleBandit.py : replication of figure 2.2

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Bandits import KArmedBandit
from IRL.agents.ActionValue import ActionValueLearner

nExperiments = 2000
nBandits = 10
nEpisodes = 1000
epsilons = [0.0, 0.01, 0.1]

avg_rewards_all = []
sum_optimalActions_all =[]
for epsilon in epsilons:

  avg_rewards = np.zeros([nEpisodes])
  sum_optimalActions = np.zeros([nEpisodes])
  for idxExperiment in range(nExperiments):
    
    print("epsilon :", epsilon, "experiment :", idxExperiment)
    
    env = KArmedBandit(nBandits)
    agent = ActionValueLearner(env.nStates, env.nActions, actionSelectionMethod="egreedy", epsilon=epsilon)

    list_rewards = []
    list_actions = []
    state = 0
    for e in range(nEpisodes):
      action = agent.selectAction(state)
      reward = env.step(action)
      agent.update(state, action, reward)

      #print("Episode : ", e, "Selected action : ", action, " Reward : ", reward)
      
      list_rewards.append(reward)
      list_actions.append(action)
      
    avg_rewards = avg_rewards + (1.0/(idxExperiment+1))*(list_rewards - avg_rewards)
    sum_optimalActions = sum_optimalActions + np.array(np.array(list_actions)==np.argmax(env.rewardMeans), dtype=np.float)
  
  avg_rewards_all.append(avg_rewards)
  sum_optimalActions_all.append(sum_optimalActions)
  
pl.figure()
for i, epsilon in enumerate(epsilons):
  pl.plot(100*avg_rewards_all[i], label="epsilon="+str(np.round(epsilon,2)))
pl.xlabel("Steps")
pl.ylabel("Average Reward")
pl.legend()

pl.figure()
for i, epsilon in enumerate(epsilons):
  pl.plot(100*sum_optimalActions_all[i]/nExperiments, label="epsilon="+str(np.round(epsilon,2)))
pl.xlabel("Steps")
pl.ylabel("% Optimal Action")
pl.legend()
pl.show()