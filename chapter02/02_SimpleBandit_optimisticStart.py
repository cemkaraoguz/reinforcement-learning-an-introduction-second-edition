'''
02_SimpleBandit_optimisticStart.py : replication of figure 2.3

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Bandits import KArmedBandit
from IRL.agents.ActionValue import ActionValueLearner

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
  nEpisodes = 1000
  alpha = 0.1
  epsilon_egreedy = 0.1
  epsilon_optimistic = 0.0
  qboost_egreedy = 0.0
  qboost_optimistic = 5.0

  env = KArmedBandit(nBandits)
  agent_egreedy = ActionValueLearner(env.nStates, env.nActions, alpha=alpha, qboost=qboost_egreedy,
    actionSelectionMethod="egreedy", epsilon=epsilon_egreedy)
  agent_optimistic = ActionValueLearner(env.nStates, env.nActions, alpha=alpha, qboost=qboost_optimistic,
    actionSelectionMethod="greedy", epsilon=epsilon_optimistic)

  avg_rewards_egreedy = np.zeros([nEpisodes])
  sum_optimalActions_egreedy = np.zeros([nEpisodes])
  avg_rewards_optimistic = np.zeros([nEpisodes])
  sum_optimalActions_optimistic = np.zeros([nEpisodes])
  for idxExperiment in range(nExperiments):
    
    print("experiment :", idxExperiment)
    
    env.reset()
    agent_egreedy.reset()
    rewards_egreedy, actions_egreedy = runExperiment(nEpisodes, env, agent_egreedy)
    avg_rewards_egreedy = avg_rewards_egreedy + (1.0/(idxExperiment+1))*(rewards_egreedy - avg_rewards_egreedy)
    sum_optimalActions_egreedy += np.array(np.array(actions_egreedy)==np.argmax(env.rewardMeans), dtype=float)

    agent_optimistic.reset()
    rewards_optimistic, actions_optimistic = runExperiment(nEpisodes, env, agent_optimistic)
    avg_rewards_optimistic = avg_rewards_optimistic + (1.0/(idxExperiment+1))*(rewards_optimistic - avg_rewards_optimistic)
    sum_optimalActions_optimistic += np.array(np.array(actions_optimistic)==np.argmax(env.rewardMeans), dtype=float)

  pl.figure()
  pl.plot(avg_rewards_egreedy, label="e-greedy, epsilon="+str(np.round(epsilon_egreedy,2)))
  pl.plot(avg_rewards_optimistic, label="Optimistic, epsilon="+str(np.round(epsilon_optimistic,2)))
  pl.xlabel("Steps")
  pl.ylabel("Average Reward")
  pl.legend()

  pl.figure()
  pl.plot(sum_optimalActions_egreedy/nExperiments*100, label="e-greedy, epsilon="+str(np.round(epsilon_egreedy,2)))
  pl.plot(sum_optimalActions_optimistic/nExperiments*100, label="Optimistic, epsilon="+str(np.round(epsilon_optimistic,2)))
  pl.xlabel("Steps")
  pl.ylabel("% Optimal Action")
  pl.legend()
  pl.show()