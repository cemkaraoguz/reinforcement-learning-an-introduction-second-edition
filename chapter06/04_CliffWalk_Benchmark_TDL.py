'''
04_CliffWalk_Benchmark_TDL.py : Benchmark of various TD Learning algorithms on Example 6.6

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.TemporalDifferenceLearning import SARSA, QLearning, ExpectedSARSA, DoubleQLearning
from IRL.utils.Helpers import runSimulation

def runExperiment(nEpisodes, env, agent):
  reward_sums = []
  for e in range(nEpisodes):
    
    if(e%50==0):
      print("Episode : ", e)
      
    state = env.reset()
    action = agent.selectAction(state)      
    done = False
    reward_sums.append(0.0)
    while not done:
    
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
      
      reward_sums[-1] += reward
      
  return reward_sums
  
if __name__=="__main__":

  nExperiments = 100
  nEpisodes = 500

  # Environment
  sizeX = 12
  sizeY = 4
  defaultReward = -1.0
  startStates = [(0,3)]
  terminalStates= [(11,3)]
  specialRewards = {((1,2),1):-100.0,((2,2),1):-100.0,((3,2),1):-100.0,((4,2),1):-100.0,((5,2),1):-100.0,
    ((6,2),1):-100.0,((7,2),1):-100.0,((8,2),1):-100.0,((9,2),1):-100.0,((10,2),1):-100.0, ((0,3),2):-100.0}

  # Agents
  alpha_SARSA = 0.5
  gamma_SARSA = 1.0
  alpha_QLearning = 0.5
  gamma_QLearning = 1.0 
  alpha_ExpectedSARSA = 0.5
  gamma_ExpectedSARSA = 1.0
  alpha_DoubleQLearning = 0.5
  gamma_DoubleQLearning = 1.0
  
  # Policy
  epsilon_SARSA = 0.1
  epsilon_QLearning = 0.1
  epsilon_ExpectedSARSA = 0.1
  epsilon_DoubleQLearning = 0.1

  avg_reward_sums_SARSA = np.zeros(nEpisodes)
  avg_reward_sums_QLearning = np.zeros(nEpisodes)
  avg_reward_sums_ExpectedSARSA = np.zeros(nEpisodes)
  avg_reward_sums_DoubleQLearning = np.zeros(nEpisodes)
  for idx_experiment in range(1, nExperiments+1):
  
    print("Experiment : ", idx_experiment)
      
    env = DeterministicGridWorld(sizeX, sizeY, specialRewards=specialRewards, defaultReward=defaultReward,
      terminalStates=terminalStates, startStates=startStates)
    
    agent_SARSA = SARSA(env.nStates, env.nActions, alpha_SARSA, gamma_SARSA, epsilon=epsilon_SARSA)
    agent_QLearning = QLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, epsilon=epsilon_QLearning)
    agent_ExpectedSARSA = ExpectedSARSA(env.nStates, env.nActions, alpha_ExpectedSARSA, gamma_ExpectedSARSA, epsilon=epsilon_ExpectedSARSA)
    agent_DoubleQLearning = DoubleQLearning(env.nStates, env.nActions, alpha_DoubleQLearning, gamma_DoubleQLearning, epsilon=epsilon_DoubleQLearning)
        
    reward_sums_SARSA = runExperiment(nEpisodes, env, agent_SARSA)
    reward_sums_QLearning = runExperiment(nEpisodes, env, agent_QLearning)
    reward_sums_ExpectedSARSA = runExperiment(nEpisodes, env, agent_ExpectedSARSA)
    reward_sums_DoubleQLearning = runExperiment(nEpisodes, env, agent_DoubleQLearning)
    
    avg_reward_sums_SARSA = avg_reward_sums_SARSA + (1.0/idx_experiment)*(reward_sums_SARSA - avg_reward_sums_SARSA)
    avg_reward_sums_QLearning = avg_reward_sums_QLearning + (1.0/idx_experiment)*(reward_sums_QLearning - avg_reward_sums_QLearning)
    avg_reward_sums_ExpectedSARSA = avg_reward_sums_ExpectedSARSA + (1.0/idx_experiment)*(reward_sums_ExpectedSARSA - avg_reward_sums_ExpectedSARSA)
    avg_reward_sums_DoubleQLearning = avg_reward_sums_DoubleQLearning + (1.0/idx_experiment)*(reward_sums_DoubleQLearning - avg_reward_sums_DoubleQLearning)
  
  pl.figure()
  pl.plot(avg_reward_sums_SARSA, '-b', label='SARSA')
  pl.plot(avg_reward_sums_QLearning, '-r', label='Q-Learning')
  pl.plot(avg_reward_sums_ExpectedSARSA, '-g', label='Expected SARSA')
  pl.plot(avg_reward_sums_DoubleQLearning, '-k', label='Double Q-Learning')
  pl.xlabel("Episodes")
  pl.ylabel("Sum of reward during episodes")
  pl.legend() 
  pl.show()
  
  agents = [agent_SARSA, agent_QLearning, agent_ExpectedSARSA, agent_DoubleQLearning]
  for agent in agents:
    print("Policy for :", agent.getName())
    env.printEnv(agent)
    input("Press any key to simulate agent "+agent.getName())
    agentHistory = runSimulation(env, agent, 200)
    print("Simulation:", agent.getName()) 
    env.render(agentHistory)