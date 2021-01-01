'''
07_CliffWalk_nStepQSigma.py : n-step Q-Sigma applied to Cliff Walking problem (Example 6.6)

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.TemporalDifferenceLearning import nStepQSigma, nStepSARSA, nStepTreeBackup
from IRL.utils.Policies import StochasticPolicy
from IRL.utils.Helpers import runSimulation

def runExperiment(nEpisodes, env, agent, policy_behaviour=None, doUpdateBehaviourPolicy=True):
  reward_sums = []
  for e in range(nEpisodes):
    
    if(e%10==0):
      print("Episode : ", e)
      
    state = env.reset()
    if policy_behaviour is None:
      action = agent.selectAction(state)
    else:
       action = policy_behaviour.sampleAction(state)
    done = False
    experiences = [{}]
    reward_sums.append(0.0)
    while not done:

      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done
      
      new_state, reward, done = env.step(action)
      
      #print("Episode:", e, "State:", state, "Action: ", env.actionMapping[action][1], "Reward: ", reward, "New state:", new_state, "done:", done)

      if policy_behaviour is None:
        new_action = agent.selectAction(state)
      else:
        new_action = policy_behaviour.sampleAction(state)
      
      xp = {}
      xp['state'] = new_state
      xp['reward'] = reward
      xp['action'] = new_action
      xp['done'] = done
      experiences.append(xp)
      
      if(policy_behaviour is None):
        agent.update(experiences[-2:])
      else:
        agent.update(experiences[-2:], policy_behaviour)
    
      state = new_state
      action = new_action
      
      reward_sums[-1] += reward

      if(policy_behaviour is not None and doUpdateBehaviourPolicy):
        # update behaviour policy to be e-soft version of the target policy
        for idx_state in range(env.nStates):
          policy_behaviour.update(idx_state, agent.actionValueTable[idx_state,:])
        
  return reward_sums
  
if __name__=="__main__":

  nExperiments = 1
  nEpisodes = 800

  # Environment
  sizeX = 12
  sizeY = 4
  defaultReward = -1.0
  startStates = [(0,3)]
  terminalStates= [(11,3)]
  specialRewards = {((1,2),1):-100.0,((2,2),1):-100.0,((3,2),1):-100.0,((4,2),1):-100.0,((5,2),1):-100.0,
    ((6,2),1):-100.0,((7,2),1):-100.0,((8,2),1):-100.0,((9,2),1):-100.0,((10,2),1):-100.0, ((0,3),2):-100.0}  


  # Agent
  alpha_nStepQSigma_1 = 0.1
  gamma_nStepQSigma_1 = 1.0
  n_nStepQSigma_1 = 10
  sigma_nStepQSigma_1 = 0.0
  
  alpha_nStepQSigma_2 = 0.1
  gamma_nStepQSigma_2 = 1.0
  n_nStepQSigma_2 = 10
  sigma_nStepQSigma_2 = 0.25
  
  alpha_nStepQSigma_3 = 0.1
  gamma_nStepQSigma_3 = 1.0
  n_nStepQSigma_3 = 10
  sigma_nStepQSigma_3 = 1.0

  alpha_nStepSARSA_1 = 0.1
  gamma_nStepSARSA_1 = 1.0
  n_nStepSARSA_1 = 10

  alpha_nStepTB_1 = 0.1
  gamma_nStepTB_1 = 1.0
  n_nStepTB_1 = 10
  
  # Policy
  doUpdateBehaviourPolicy = True
  epsilon_behaviourPolicy = 0.4
  epsilon_nStepSARSA_1 = 0.1
  
  env = DeterministicGridWorld(sizeX, sizeY, specialRewards=specialRewards, defaultReward=defaultReward,
    terminalStates=terminalStates, startStates=startStates)

  env.printEnv()

  avg_reward_sums_nStepQSigma_1 = np.zeros(nEpisodes)
  avg_reward_sums_nStepQSigma_2 = np.zeros(nEpisodes)
  avg_reward_sums_nStepQSigma_3 = np.zeros(nEpisodes)
  avg_reward_sums_nStepSARSA_1 = np.zeros(nEpisodes)
  avg_reward_sums_nStepTB_1 = np.zeros(nEpisodes)
  for idx_experiment in range(1, nExperiments+1):
  
    print("Experiment : ", idx_experiment)
    
    agent_nStepQSigma_1 = nStepQSigma(env.nStates, env.nActions, alpha_nStepQSigma_1, gamma_nStepQSigma_1, n_nStepQSigma_1, sigma_nStepQSigma_1)
    agent_nStepQSigma_2 = nStepQSigma(env.nStates, env.nActions, alpha_nStepQSigma_2, gamma_nStepQSigma_2, n_nStepQSigma_2, sigma_nStepQSigma_2)
    agent_nStepQSigma_3 = nStepQSigma(env.nStates, env.nActions, alpha_nStepQSigma_3, gamma_nStepQSigma_3, n_nStepQSigma_3, sigma_nStepQSigma_3)
    agent_nStepSARSA_1 = nStepSARSA(env.nStates, env.nActions, alpha_nStepSARSA_1, gamma_nStepSARSA_1, n_nStepSARSA_1, epsilon=epsilon_nStepSARSA_1)
    agent_nStepTB_1 = nStepTreeBackup(env.nStates, env.nActions, alpha_nStepTB_1, gamma_nStepTB_1, n_nStepTB_1)
    
    policy_behaviour = StochasticPolicy(env.nStates, env.nActions, policyUpdateMethod="esoft", epsilon=epsilon_behaviourPolicy)
    reward_sums_nStepQSigma_1 = runExperiment(nEpisodes, env, agent_nStepQSigma_1, policy_behaviour, doUpdateBehaviourPolicy)

    policy_behaviour = StochasticPolicy(env.nStates, env.nActions, policyUpdateMethod="esoft", epsilon=epsilon_behaviourPolicy) 
    reward_sums_nStepQSigma_2 = runExperiment(nEpisodes, env, agent_nStepQSigma_2, policy_behaviour, doUpdateBehaviourPolicy)

    policy_behaviour = StochasticPolicy(env.nStates, env.nActions, policyUpdateMethod="esoft", epsilon=epsilon_behaviourPolicy)
    reward_sums_nStepQSigma_3 = runExperiment(nEpisodes, env, agent_nStepQSigma_3, policy_behaviour, doUpdateBehaviourPolicy)

    reward_sums_nStepSARSA_1 = runExperiment(nEpisodes, env, agent_nStepSARSA_1)

    policy_behaviour = StochasticPolicy(env.nStates, env.nActions, policyUpdateMethod="esoft", epsilon=epsilon_behaviourPolicy)
    reward_sums_nStepTB_1 = runExperiment(nEpisodes, env, agent_nStepTB_1, policy_behaviour, doUpdateBehaviourPolicy)
    
    avg_reward_sums_nStepQSigma_1 = avg_reward_sums_nStepQSigma_1 + (1.0/idx_experiment)*(reward_sums_nStepQSigma_1 - avg_reward_sums_nStepQSigma_1)
    avg_reward_sums_nStepQSigma_2 = avg_reward_sums_nStepQSigma_2 + (1.0/idx_experiment)*(reward_sums_nStepQSigma_2 - avg_reward_sums_nStepQSigma_2)
    avg_reward_sums_nStepQSigma_3 = avg_reward_sums_nStepQSigma_3 + (1.0/idx_experiment)*(reward_sums_nStepQSigma_3 - avg_reward_sums_nStepQSigma_3)
    avg_reward_sums_nStepSARSA_1 = avg_reward_sums_nStepSARSA_1 + (1.0/idx_experiment)*(reward_sums_nStepSARSA_1 - avg_reward_sums_nStepSARSA_1)
    avg_reward_sums_nStepTB_1 = avg_reward_sums_nStepTB_1 + (1.0/idx_experiment)*(reward_sums_nStepTB_1 - avg_reward_sums_nStepTB_1)
  
  pl.figure()
  pl.plot(avg_reward_sums_nStepQSigma_1, '-r', label=str(n_nStepQSigma_1)+' Step Q-Sigma, sigma='+str(sigma_nStepQSigma_1))
  pl.plot(avg_reward_sums_nStepQSigma_2, '-g', label=str(n_nStepQSigma_2)+' Step Q-Sigma, sigma='+str(sigma_nStepQSigma_2))
  pl.plot(avg_reward_sums_nStepQSigma_3, '-k', label=str(n_nStepQSigma_3)+' Step Q-Sigma, sigma='+str(sigma_nStepQSigma_3))
  pl.xlabel("Episodes")
  pl.ylabel("Sum of reward during episodes")
  pl.legend() 
  pl.figure()
  pl.plot(avg_reward_sums_nStepQSigma_3, '-r', label=str(n_nStepQSigma_3)+' Step Q-Sigma, sigma='+str(sigma_nStepQSigma_3))
  pl.plot(avg_reward_sums_nStepSARSA_1, '--k', label=str(n_nStepSARSA_1)+' Step SARSA')
  pl.xlabel("Episodes")
  pl.ylabel("Sum of reward during episodes")
  pl.legend() 
  pl.figure()
  pl.plot(avg_reward_sums_nStepQSigma_1, '-r', label=str(n_nStepQSigma_1)+' Step Q-Sigma, sigma='+str(sigma_nStepQSigma_1))
  pl.plot(avg_reward_sums_nStepTB_1, '--k', label=str(n_nStepTB_1)+' Step Tree Backup')
  pl.xlabel("Episodes")
  pl.ylabel("Sum of reward during episodes")
  pl.legend() 
  pl.show()
  
  agents = [agent_nStepQSigma_1, agent_nStepQSigma_2, agent_nStepQSigma_3]
  for agent in agents:
    print("Policy for :", agent.getName())
    env.printEnv(agent)

  for agent in agents:
    input("Press any key to simulate agent "+agent.getName())
    agentHistory = runSimulation(env, agent, 200)
    print("Simulation:", agent.getName()) 
    env.render(agentHistory)
    