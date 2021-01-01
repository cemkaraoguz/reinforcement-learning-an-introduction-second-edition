'''
02_WindyGridWorld_nStepSARSA.py : n-step SARSA applied to Windy Grid World problem (Example 6.5)

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import StochasticGridWorld
from IRL.agents.TemporalDifferenceLearning import SARSA, nStepSARSA
from IRL.utils.Helpers import runSimulation

def runExperiment(nEpisodes, env, agent):
  reward_sums = []
  episodesvstimesteps = []
  timesteps = 0
  for e in range(nEpisodes):
    
    if(e%10==0):
      print("Episode : ", e)
      
    experiences = [{}]
    state = env.reset()
    action = agent.selectAction(state)  
    done = False
    reward_sums.append(0.0)
    while not done:

      timesteps += 1
      
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
    
      state = new_state
      action = new_action
      
      episodesvstimesteps.append([e,timesteps])
      reward_sums[-1] += reward
      
  return reward_sums, np.array(episodesvstimesteps)
  
if __name__=="__main__":

  exerciseID = 0
  nExperiments = 1
  nEpisodes = 800

  # Environment
  sizeX = 10
  sizeY = 7
  defaultReward = -1.0
  startStates = [(0,3)]
  terminalStates= [(7,3)]

  if exerciseID==0:
    # Example 6.5
    actionMapping = {0:(np.array([0,-1]), "N"), 1:(np.array([0,1]), "S"), 2:(np.array([1,0]), "E"), 3:(np.array([-1,0]), "W")}
    sigmaY_actionNoise = 0
  
  elif exerciseID==1:
    # Exercise 6.9 part 1
    actionMapping = {0:(np.array([0,-1]), "N"), 1:(np.array([0,1]), "S"), 2:(np.array([1,0]), "E"), 3:(np.array([-1,0]), "W"),
      4:(np.array([1,-1]), "NE"), 5:(np.array([1,1]), "SE"), 6:(np.array([-1,-1]), "NW"), 7:(np.array([-1,1]), "SW")}
    
    # Example 6.5 and Exercise 6.9
    sigmaY_actionNoise = 0
    
    # Exercise 6.10
    sigmaY_actionNoise = 1
    
  else:
    # Exercise 6.9 part 2
    actionMapping = {0:(np.array([0,-1]), "N"), 1:(np.array([0,1]), "S"), 2:(np.array([1,0]), "E"), 3:(np.array([-1,0]), "W"),
      4:(np.array([1,-1]), "NE"), 5:(np.array([1,1]), "SE"), 6:(np.array([-1,-1]), "NW"), 7:(np.array([-1,1]), "SW"), 8:(np.array([0,0]), "0")}
    sigmaY_actionNoise = 0

  actionNoiseParams = {}
  aux = [(x,y) for x in range(3,6) for y in range(0,7)]
  for pos in aux:
    actionNoiseParams[pos] = [0,-1,0,sigmaY_actionNoise]
  aux = [(x,y) for x in range(6,8) for y in range(0,7)]
  for pos in aux:
    actionNoiseParams[pos] = [0,-2,0,sigmaY_actionNoise]
  aux = [(8,y) for y in range(0,7)]
  for pos in aux:
    actionNoiseParams[pos] = [0,-1,0,sigmaY_actionNoise]
    
  # Agent
  alpha_SARSA = 0.1
  gamma_SARSA = 1.0
  alpha_nStepSARSA_1 = 0.2
  gamma_nStepSARSA_1 = 1.0
  n_nStepSARSA_1 = 1
  alpha_nStepSARSA_2 = 0.2
  gamma_nStepSARSA_2 = 1.0
  n_nStepSARSA_2 = 5  
  alpha_nStepSARSA_3 = 0.2
  gamma_nStepSARSA_3 = 1.0
  n_nStepSARSA_3 = 30
  
  # Policy
  epsilon_SARSA = 0.1
  epsilon_nStepSARSA_1 = 0.1
  epsilon_nStepSARSA_2 = 0.1
  epsilon_nStepSARSA_3 = 0.1
  
  env = StochasticGridWorld(sizeX, sizeY, actionNoiseParams=actionNoiseParams, startStates=startStates,
    defaultReward=defaultReward, terminalStates=terminalStates, actionMapping=actionMapping)

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
    
    reward_sums_SARSA, evst_SARSA = runExperiment(nEpisodes, env, agent_SARSA)
    reward_sums_nStepSARSA_1, evst_nStepSARSA_1 = runExperiment(nEpisodes, env, agent_nStepSARSA_1)
    reward_sums_nStepSARSA_2, evst_nStepSARSA_2 = runExperiment(nEpisodes, env, agent_nStepSARSA_2)
    reward_sums_nStepSARSA_3, evst_nStepSARSA_3 = runExperiment(nEpisodes, env, agent_nStepSARSA_3)
    
    avg_reward_sums_SARSA = avg_reward_sums_SARSA + (1.0/idx_experiment)*(reward_sums_SARSA - avg_reward_sums_SARSA)
    avg_reward_sums_nStepSARSA_1 = avg_reward_sums_nStepSARSA_1 + (1.0/idx_experiment)*(reward_sums_nStepSARSA_1 - avg_reward_sums_nStepSARSA_1)
    avg_reward_sums_nStepSARSA_2 = avg_reward_sums_nStepSARSA_2 + (1.0/idx_experiment)*(reward_sums_nStepSARSA_2 - avg_reward_sums_nStepSARSA_2)
    avg_reward_sums_nStepSARSA_3 = avg_reward_sums_nStepSARSA_3 + (1.0/idx_experiment)*(reward_sums_nStepSARSA_3 - avg_reward_sums_nStepSARSA_3)
    
  pl.figure()
  pl.plot(evst_SARSA[:,1],evst_SARSA[:,0], '-b', label='SARSA')
  pl.plot(evst_nStepSARSA_1[:,1],evst_nStepSARSA_1[:,0], '-r', label=str(n_nStepSARSA_1)+' Step SARSA')
  pl.plot(evst_nStepSARSA_2[:,1],evst_nStepSARSA_2[:,0], '-g', label=str(n_nStepSARSA_2)+' Step SARSA')
  pl.plot(evst_nStepSARSA_3[:,1],evst_nStepSARSA_3[:,0], '-k', label=str(n_nStepSARSA_3)+' Step SARSA')
  pl.xlabel("Time steps")
  pl.ylabel("Episodes")
  pl.legend() 
  
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

  for agent in agents:
    input("Press any key to simulate agent "+agent.getName())
    agentHistory = runSimulation(env, agent, 200)
    print("Simulation:", agent.getName()) 
    env.render(agentHistory)