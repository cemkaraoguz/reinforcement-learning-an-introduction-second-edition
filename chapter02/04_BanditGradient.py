'''
04_BanditGradient.py : replication of figure 2.4

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Bandits import KArmedBandit
from IRL.agents.PolicyGradient import BanditGradient

def runExperiment(nEpisodes, env, agent):
  rewards = np.zeros([nEpisodes])
  actions = np.zeros([nEpisodes])
  state = 0
  for e in range(nEpisodes):
    action = agent.selectAction(state)
    reward = env.step(action)
    agent.update(state, action, reward)

    #print("Episode : ", e, "Selected action : ", action, " Reward : ", reward)
    
    rewards[e] = reward
    actions[e] = action
    
  return rewards, actions

if __name__=="__main__":
  nExperiments = 2000
  nBandits = 10
  rewardDist_mean = 4.0
  nEpisodes = 1000
  alphas = [0.1, 0.4, 0.1, 0.4]
  baselines = [False, False, True, True]

  env = KArmedBandit(nBandits, rewardDist_mean=rewardDist_mean)
  agents = []
  for alpha, doUseBaseline in zip(alphas, baselines):
    agent = BanditGradient(env.nStates, env.nActions, alpha, doUseBaseline)
    agents.append(agent)

  avg_rewards_all = []
  sum_optimalActions_all = []
  for agent in agents:
    avg_rewards = np.zeros([nEpisodes])
    sum_optimalActions = np.zeros([nEpisodes])
    for idxExperiment in range(nExperiments):
      
      if idxExperiment%100==0: print("experiment :", idxExperiment)
      
      env.reset()
      agent.reset()
      rewards, actions = runExperiment(nEpisodes, env, agent)
      avg_rewards = avg_rewards + (1.0/(idxExperiment+1))*(rewards - avg_rewards)
      sum_optimalActions += np.array(np.array(actions)==np.argmax(env.rewardMeans), dtype=float)

    avg_rewards_all.append(avg_rewards)
    sum_optimalActions_all.append(sum_optimalActions)
    
  pl.figure()
  for i, agent in enumerate(agents):
    pl.plot(avg_rewards_all[i], label="alpha="+str(np.round(agent.alpha,2))+" baseline:"+str(agent.doUseBaseline))
  pl.xlabel("Steps")
  pl.ylabel("Average Reward")
  pl.legend()

  pl.figure()
  for i, agent in enumerate(agents):
    pl.plot(sum_optimalActions_all[i]/nExperiments*100, label="alpha="+str(np.round(agent.alpha,2))+" baseline:"+str(agent.doUseBaseline))
  pl.xlabel("Steps")
  pl.ylabel("% Optimal Action")
  pl.legend()
  pl.show()