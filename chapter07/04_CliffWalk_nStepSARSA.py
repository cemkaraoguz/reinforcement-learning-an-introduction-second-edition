'''
04_CliffWalk_nStepSARSA.py : n-step SARSA applied to Cliff Walking problem (Example 6.6)

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.TemporalDifferenceLearning import SARSA, nStepSARSA
from IRL.utils.Helpers import runSimulation

def runExperiment(nEpisodes, env, agent):
  reward_sums = []
  for e in range(nEpisodes):
    
    if(e%10==0):
      print("Episode : ", e)
      
    state = env.reset()
    action = agent.selectAction(state)    
    done = False
    experiences = [{}]
    reward_sums.append(0.0)
    while not done:
      
      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done
      
      new_state, reward, done = env.step(action)
      
      #print("State:", state, "Action: ", env.actionMapping[action][1], "Reward: ", reward, "New state:", new_state, "done:", done)
      
      new_action = agent.selectAction(new_state)
      
      xp = {}
      xp['state'] = new_state
      xp['reward'] = reward
      xp['done'] = done
      xp['action'] = new_action
      experiences.append(xp)
      
      agent.update(experiences[-2:])
      
      if(agent.getName()=="SARSA"):
        action = new_action
      else:
        action = agent.selectAction(new_state)
      
      state = new_state
      
      reward_sums[-1] += reward
      
  return reward_sums
  
if __name__=="__main__":

  nExperiments = 1
  nEpisodes = 400

  # Environment
  sizeX = 12
  sizeY = 4
  defaultReward = -1.0
  startStates = [(0,3)]
  terminalStates= [(11,3)]
  specialRewards = {((1,2),1):-100.0,((2,2),1):-100.0,((3,2),1):-100.0,((4,2),1):-100.0,((5,2),1):-100.0,
    ((6,2),1):-100.0,((7,2),1):-100.0,((8,2),1):-100.0,((9,2),1):-100.0,((10,2),1):-100.0, ((0,3),2):-100.0}
  
  # Agent
  alpha_SARSA = 0.5
  gamma_SARSA = 1.0
  alpha_nStepSARSA_1 = 0.5
  gamma_nStepSARSA_1 = 1.0
  n_nStepSARSA_1 = 1
  alpha_nStepSARSA_2 = 0.05
  gamma_nStepSARSA_2 = 1.0
  n_nStepSARSA_2 = 5  
  alpha_nStepSARSA_3 = 0.01
  gamma_nStepSARSA_3 = 1.0
  n_nStepSARSA_3 = 15
  
  # Policy
  epsilon_SARSA = 0.1
  epsilon_nStepSARSA_1 = 0.1
  epsilon_nStepSARSA_2 = 0.1
  epsilon_nStepSARSA_3 = 0.1
  
  env = DeterministicGridWorld(sizeX, sizeY, specialRewards=specialRewards, defaultReward=defaultReward,
    terminalStates=terminalStates, startStates=startStates)

  env.printEnv()

  avg_reward_sums_SARSA = np.zeros(nEpisodes)
  avg_reward_sums_nStepSARSA_1 = np.zeros(nEpisodes)
  avg_reward_sums_nStepSARSA_2 = np.zeros(nEpisodes)
  avg_reward_sums_nStepSARSA_3 = np.zeros(nEpisodes)
  for idx_experiment in range(1, nExperiments+1):
  
    print("Experiment : ", idx_experiment)
    
    agent_SARSA = SARSA(env.nStates, env.nActions, alpha_SARSA, gamma_SARSA, epsilon=epsilon_SARSA)
    agent_nStepSARSA_1 = nStepSARSA(env.nStates, env.nActions, alpha_nStepSARSA_1, gamma_nStepSARSA_1, n_nStepSARSA_1, epsilon=epsilon_nStepSARSA_1)
    agent_nStepSARSA_2 = nStepSARSA(env.nStates, env.nActions, alpha_nStepSARSA_2, gamma_nStepSARSA_2, n_nStepSARSA_2, epsilon=epsilon_nStepSARSA_2)
    agent_nStepSARSA_3 = nStepSARSA(env.nStates, env.nActions, alpha_nStepSARSA_3, gamma_nStepSARSA_3, n_nStepSARSA_3, epsilon=epsilon_nStepSARSA_3)
    
    reward_sums_SARSA = runExperiment(nEpisodes, env, agent_SARSA)
    reward_sums_nStepSARSA_1 = runExperiment(nEpisodes, env, agent_nStepSARSA_1)
    reward_sums_nStepSARSA_2 = runExperiment(nEpisodes, env, agent_nStepSARSA_2)
    reward_sums_nStepSARSA_3 = runExperiment(nEpisodes, env, agent_nStepSARSA_3)
    
    avg_reward_sums_SARSA = avg_reward_sums_SARSA + (1.0/idx_experiment)*(reward_sums_SARSA - avg_reward_sums_SARSA)
    avg_reward_sums_nStepSARSA_1 = avg_reward_sums_nStepSARSA_1 + (1.0/idx_experiment)*(reward_sums_nStepSARSA_1 - avg_reward_sums_nStepSARSA_1)
    avg_reward_sums_nStepSARSA_2 = avg_reward_sums_nStepSARSA_2 + (1.0/idx_experiment)*(reward_sums_nStepSARSA_2 - avg_reward_sums_nStepSARSA_2)
    avg_reward_sums_nStepSARSA_3 = avg_reward_sums_nStepSARSA_3 + (1.0/idx_experiment)*(reward_sums_nStepSARSA_3 - avg_reward_sums_nStepSARSA_3)
  
  pl.figure()
  pl.plot(avg_reward_sums_SARSA, '-b', label='SARSA')
  pl.plot(avg_reward_sums_nStepSARSA_1, '-r', label=str(n_nStepSARSA_1)+' Step SARSA')
  pl.plot(avg_reward_sums_nStepSARSA_2, '-g', label=str(n_nStepSARSA_2)+' Step SARSA')
  pl.plot(avg_reward_sums_nStepSARSA_3, '-k', label=str(n_nStepSARSA_3)+' Step SARSA')
  pl.xlabel("Episodes")
  pl.ylabel("Sum of reward during episodes")
  pl.legend() 
  pl.show()
  
  agents = [agent_SARSA, agent_nStepSARSA_1, agent_nStepSARSA_2, agent_nStepSARSA_3]
  for agent in agents:
    print("Policy for :", agent.getName())
    env.printEnv(agent)
    input("Press any key to simulate agent "+agent.getName())
    agentHistory = runSimulation(env, agent, 100)
    print("Simulation:", agent.getName()) 
    env.render(agentHistory)