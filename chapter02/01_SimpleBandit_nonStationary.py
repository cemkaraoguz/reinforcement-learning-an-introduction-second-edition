'''
01_SimpleBandit_nonStationary.py : solution to exercise 2.5

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Bandits import KArmedBandit
from IRL.agents.ActionValue import ActionValueLearner

nExperiments = 2000
nBandits = 10
epsilon = 0.1
nEpisodes = 10000
doUseSampleAverages = False
envRewardNoise_mean = 0.0
envRewardNoise_std = 0.01

if(doUseSampleAverages):
  alpha = None
  experimentTitle = "Sample averages, epsilon="+str(epsilon)+", nExperiments="+str(nExperiments)
else:
  alpha = 0.1
  experimentTitle = "Weighted averages, epsilon="+str(epsilon)+", nExperiments="+str(nExperiments)+", alpha="+str(alpha)

avg_rewards = np.zeros([nEpisodes])
sum_optimalActions = np.zeros([nEpisodes])
for idxExperiment in range(nExperiments):

  print("Experiment :", idxExperiment)

  env = KArmedBandit(nBandits, rewardDist_mean=0.0, rewardDist_std=0.0) # rewards start out equal
  agent = ActionValueLearner(env.nStates, env.nActions, alpha=alpha, actionSelectionMethod="egreedy", epsilon=epsilon)
  list_rewards = []
  list_actions = []
  state = 0
  for e in range(nEpisodes):

    action = agent.policy.selectAction(state)
    reward = env.step(action)
    agent.update(state, action, reward)

    #print("Epoch : ", e, "Selected action : ", action, " Reward : ", reward)
    
    list_rewards.append(reward)
    list_actions.append(action)
    env.addGaussianNoiseToRewards(envRewardNoise_mean, envRewardNoise_std)
    
  optimalActionRate = (np.sum(np.array(list_actions)==np.argmax(env.rewards)))/(nEpisodes*1.0)  
  avg_rewards = avg_rewards + (1.0/(idxExperiment+1))*(list_rewards - avg_rewards)
  sum_optimalActions = sum_optimalActions + np.array(np.array(list_actions)==np.argmax(env.rewards), dtype=np.float)
  
pl.figure()
pl.plot(avg_rewards)
pl.xlabel("Steps")
pl.ylabel("Average Reward")
pl.title(experimentTitle)

pl.figure()
pl.plot(100.0*sum_optimalActions/nExperiments)
pl.xlabel("Steps")
pl.ylabel("% Optimal Action")
pl.title(experimentTitle)

pl.show()