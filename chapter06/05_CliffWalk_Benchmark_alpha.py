'''
05_CliffWalk_Benchmark_alpha.py : replication of Figure 6.3

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.TemporalDifferenceLearning import SARSA, QLearning, ExpectedSARSA, DoubleQLearning

def runExperiment(nEpisodes, env, agent):
  reward_sums = []
  episodesvstimesteps = []
  timesteps = 0
  for e in range(nEpisodes):
    
    if(e%1000==0):
      print("Episode : ", e)
      
    state = env.reset()
    action = agent.selectAction(state)  
    done = False
    reward_sums.append(0.0)
    while not done:

      timesteps += 1
      
      experiences = [{}]
      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done
      
      new_state, reward, done = env.step(action)
      
      #print("State:", state, "Action: ", env.actionMapping[action][1], "Reward: ", reward, "New state:", new_state)
      
      new_action = agent.selectAction(new_state)
      
      xp = {}
      xp['reward'] = reward
      xp['state'] = new_state
      xp['done'] = done
      xp['action'] = new_action
      experiences.append(xp)

      agent.update(experiences[-2:])
    
      state = new_state
      
      if(agent.getName()=="SARSA"):
        action = new_action
      else:
        action = agent.selectAction(state)
      
      episodesvstimesteps.append([e,timesteps])
      reward_sums[-1] += reward
      
  return reward_sums, np.array(episodesvstimesteps)
  
if __name__=="__main__":

  nExperimentsList = [5, 1]
  nEpisodesList = [100, 10000]

  # Environment
  sizeX = 12
  sizeY = 4
  defaultReward = -1.0
  startStates = [(0,3)]
  terminalStates= [(11,3)]
  specialRewards = {((1,2),1):-100.0,((2,2),1):-100.0,((3,2),1):-100.0,((4,2),1):-100.0,((5,2),1):-100.0,
    ((6,2),1):-100.0,((7,2),1):-100.0,((8,2),1):-100.0,((9,2),1):-100.0,((10,2),1):-100.0, ((0,3),2):-100.0}
    
  # Agent
  alphas = np.arange(0.1,1.01,0.1)
  gamma_SARSA = 1.0
  gamma_QLearning = 1.0 
  gamma_ExpectedSARSA = 1.0
  gamma_DoubleQLearning = 1.0 
  
  # Policy
  epsilon_SARSA = 0.1
  epsilon_QLearning = 0.1
  epsilon_ExpectedSARSA = 0.1
  epsilon_DoubleQLearning = 0.1
  
  env = DeterministicGridWorld(sizeX, sizeY, specialRewards=specialRewards, defaultReward=defaultReward,
    terminalStates=terminalStates, startStates=startStates)

  env.printEnv()

  avg_reward_sums_SARSA = np.zeros([len(alphas),len(nEpisodesList)])
  avg_reward_sums_QLearning = np.zeros([len(alphas),len(nEpisodesList)])
  avg_reward_sums_ExpectedSARSA = np.zeros([len(alphas),len(nEpisodesList)])
  avg_reward_sums_DoubleQLearning = np.zeros([len(alphas),len(nEpisodesList)])
  for idx_alpha, alpha in enumerate(alphas):
    for idx_nEpisodes, nEpisodes in enumerate(nEpisodesList):
      
      nExperiments = nExperimentsList[idx_nEpisodes]
      
      for idx_experiment in range(1, nExperiments+1):
        
        print("Alpha:", alpha, "nEpisodes:", nEpisodes, "Experiment : ", idx_experiment)
        
        agent_SARSA = SARSA(env.nStates, env.nActions, alpha, gamma_SARSA, epsilon=epsilon_SARSA)
        agent_QLearning = QLearning(env.nStates, env.nActions, alpha, gamma_QLearning, epsilon=epsilon_QLearning)
        agent_ExpectedSARSA = ExpectedSARSA(env.nStates, env.nActions, alpha, gamma_ExpectedSARSA, epsilon=epsilon_ExpectedSARSA)
        agent_DoubleQLearning = DoubleQLearning(env.nStates, env.nActions, alpha, gamma_DoubleQLearning, epsilon=epsilon_DoubleQLearning)
        
        reward_sums_SARSA, evst_SARSA = runExperiment(nEpisodes, env, agent_SARSA)
        reward_sums_QLearning, evst_QLearning = runExperiment(nEpisodes, env, agent_QLearning)
        reward_sums_ExpectedSARSA, evst_ExpectedSARSA = runExperiment(nEpisodes, env, agent_ExpectedSARSA)
        reward_sums_DoubleQLearning, evst_DoubleQLearning = runExperiment(nEpisodes, env, agent_DoubleQLearning)
        
        avg_reward_sums_SARSA[idx_alpha, idx_nEpisodes] = avg_reward_sums_SARSA[idx_alpha, idx_nEpisodes] + (1.0/idx_experiment)*(np.sum(reward_sums_SARSA)/nEpisodes - avg_reward_sums_SARSA[idx_alpha, idx_nEpisodes])
        avg_reward_sums_QLearning[idx_alpha, idx_nEpisodes] = avg_reward_sums_QLearning[idx_alpha, idx_nEpisodes] + (1.0/idx_experiment)*(np.sum(reward_sums_QLearning)/nEpisodes - avg_reward_sums_QLearning[idx_alpha, idx_nEpisodes])
        avg_reward_sums_ExpectedSARSA[idx_alpha, idx_nEpisodes] = avg_reward_sums_ExpectedSARSA[idx_alpha, idx_nEpisodes] + (1.0/idx_experiment)*(np.sum(reward_sums_ExpectedSARSA)/nEpisodes - avg_reward_sums_ExpectedSARSA[idx_alpha, idx_nEpisodes])
        avg_reward_sums_DoubleQLearning[idx_alpha, idx_nEpisodes] = avg_reward_sums_DoubleQLearning[idx_alpha, idx_nEpisodes] + (1.0/idx_experiment)*(np.sum(reward_sums_DoubleQLearning)/nEpisodes - avg_reward_sums_DoubleQLearning[idx_alpha, idx_nEpisodes])
      
  fig, ax = pl.subplots()
  ax.plot(avg_reward_sums_SARSA[:,0], '--bv', label='SARSA (interim)')
  ax.plot(avg_reward_sums_QLearning[:,0], '--ks', label='Q-Learning (interim)')
  ax.plot(avg_reward_sums_ExpectedSARSA[:,0], '--rx', label='Expected SARSA (interim)')
  #ax.plot(avg_reward_sums_DoubleQLearning[:,0], '--g', label='Double Q-Learning (interim)')
  ax.plot(avg_reward_sums_SARSA[:,1], '-bv', label='SARSA (asymptotic)')
  ax.plot(avg_reward_sums_QLearning[:,1], '-ks', label='Q-Learning (asymptotic)')
  ax.plot(avg_reward_sums_ExpectedSARSA[:,1], '-rx', label='Expected SARSA (asymptotic)')
  #ax.plot(avg_reward_sums_DoubleQLearning[:,1], '-g', label='Double Q-Learning (asymptotic)')
  ax.set_xlabel("Alpha")
  ax.set_ylabel("Sum of rewards per episode")
  ax.set_xticks(range(len(alphas)))
  ax.set_xticklabels([str(np.round(i,1)) for i in alphas])
  pl.legend() 
  pl.show()
  
  agents = [agent_SARSA, agent_QLearning, agent_ExpectedSARSA, agent_DoubleQLearning]
  for agent in agents:
    print("Policy for :", agent.getName())
    env.printEnv(agent)
  