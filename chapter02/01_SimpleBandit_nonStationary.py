'''
01_SimpleBandit_nonStationary.py : solution to exercise 2.5

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Bandits import KArmedBandit
from IRL.agents.ActionValue import ActionValueLearner

if __name__=="__main__":  
  nExperiments = 2000
  nBandits = 10
  epsilon = 0.1
  nEpisodes = 10000
  envRewardNoise_mean = 0.0
  envRewardNoise_std = 0.01
  alpha = 0.1

  avg_rewards_sa = np.zeros([nEpisodes])
  sum_optimalActions_sa = np.zeros([nEpisodes])
  avg_rewards_wa = np.zeros([nEpisodes])
  sum_optimalActions_wa = np.zeros([nEpisodes])
  for idxExperiment in range(nExperiments):

    print("Experiment :", idxExperiment)

    env = KArmedBandit(nBandits, rewardDist_mean=0.0, rewardDist_std=0.0) # rewards start out equal
    agent_sa = ActionValueLearner(env.nStates, env.nActions, alpha=None, actionSelectionMethod="egreedy", epsilon=epsilon)
    agent_wa = ActionValueLearner(env.nStates, env.nActions, alpha=alpha, actionSelectionMethod="egreedy", epsilon=epsilon)
    
    list_rewards_sa = []
    list_actions_sa = []
    list_rewards_wa = []
    list_actions_wa = []    
    state = 0
    for e in range(nEpisodes):
      # Sample averages agent
      action = agent_sa.policy.selectAction(state)
      reward = env.step(action)
      agent_sa.update(state, action, reward)
      list_rewards_sa.append(reward)
      list_actions_sa.append(action)
      # Weighted averages agent
      action = agent_wa.policy.selectAction(state)
      reward = env.step(action)
      agent_wa.update(state, action, reward)
      list_rewards_wa.append(reward)
      list_actions_wa.append(action)
      # Add noise to the environment
      env.addGaussianNoiseToRewards(envRewardNoise_mean, envRewardNoise_std)
      
    avg_rewards_sa = avg_rewards_sa + (1.0/(idxExperiment+1))*(list_rewards_sa - avg_rewards_sa)
    sum_optimalActions_sa = sum_optimalActions_sa + np.array(np.array(list_actions_sa)==np.argmax(env.rewardMeans), dtype=np.float)
    avg_rewards_wa = avg_rewards_wa + (1.0/(idxExperiment+1))*(list_rewards_wa - avg_rewards_wa)
    sum_optimalActions_wa = sum_optimalActions_wa + np.array(np.array(list_actions_wa)==np.argmax(env.rewardMeans), dtype=np.float)
    
  pl.figure()
  pl.plot(avg_rewards_sa, label="Sample averages")
  pl.plot(avg_rewards_wa, label="Weighted averages")
  pl.xlabel("Steps")
  pl.ylabel("Average Reward")
  pl.legend()
  pl.figure()
  pl.plot(100.0*sum_optimalActions_sa/nExperiments, label="Sample averages")
  pl.plot(100.0*sum_optimalActions_wa/nExperiments, label="Weighted averages")
  pl.xlabel("Steps")
  pl.ylabel("% Optimal Action")
  pl.legend()
  pl.show()