'''
00_DynaMaze.py : Replication of Figure 8.2

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.TemporalDifferenceLearning import DynaQ, nStepSARSA, nStepQSigma, nStepTreeBackup
from IRL.utils.Policies import StochasticPolicy
from IRL.utils.Helpers import runSimulation
  
def runExperiment(nEpisodes, env, agent, policy_behaviour=None, doUpdateBehaviourPolicy=True):
  cumulative_reward = [0]
  nStepsPerEpisode = np.zeros(nEpisodes)
  for e in range(nEpisodes):
    
    if(e%10==0):
      print("Episode : ", e)
      
    state = env.reset()
    if(policy_behaviour is None):
      action = agent.selectAction(state)
    else:
      action = policy_behaviour.sampleAction(state)
    done = False
    experiences = [{}]
    while not done:
      
      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done
      
      new_state, reward, done = env.step(action)
      
      #print("Episode:", e, "State:", state, "Action: ", env.actionMapping[action][1], "Reward: ", reward, "New state:", new_state, "done:", done)
      
      if(policy_behaviour is None):
        new_action = agent.selectAction(new_state)
      else:
        new_action = policy_behaviour.sampleAction(new_state)
      
      xp = {}
      xp['reward'] = reward
      xp['state'] = new_state
      xp['action'] = new_action
      xp['done'] = done
      experiences.append(xp)
      
      if(policy_behaviour is None):
        agent.update(experiences[-2:])
      else:
        agent.update(experiences[-2:], policy_behaviour)
    
      state = new_state
      action = new_action
      
      cumulative_reward.append(cumulative_reward[-1]+reward)
      nStepsPerEpisode[e] += 1

      if(policy_behaviour is not None and doUpdateBehaviourPolicy):
        # update behaviour policy to be e-soft version of the target policy
        for idx_state in range(env.nStates):
          policy_behaviour.update(idx_state, agent.actionValueTable[idx_state,:])       
      
  return cumulative_reward, nStepsPerEpisode
  
if __name__=="__main__":

  nExperiments = 1
  nEpisodes = 50

  # Environment
  sizeX = 9
  sizeY = 6
  defaultReward = 0.0
  startStates = [(0,2)]
  terminalStates = [(8,0)]
  specialRewards = {((terminalStates[0][0],terminalStates[0][1]+1),0):1.0}
  impassableStates = [(5,4)]
  impassableStates.extend([(2,y) for y in range(1,4)])
  impassableStates.extend([(7,y) for y in range(0,3)])
  
  # Agent
  alpha_DynaQ = 0.1
  gamma_DynaQ = 0.95
  epsilon_DynaQ = 0.1
  nPlanningSteps_list = [0, 5, 50]
  
  alpha_nStepSARSA = 0.1
  gamma_nStepSARSA = 0.95
  n_nStepSARSA = 50
  
  alpha_nStepTB = 0.1
  gamma_nStepTB = 0.95
  n_nStepTB = 50  

  alpha_nStepQSigma = 0.1
  gamma_nStepQSigma = 0.95
  n_nStepQSigma = 50
  sigma_nStepQSigma = 0.5
  
  # Policy
  doUpdateBehaviourPolicy = True
  epsilon_behaviourPolicy = 0.4
  epsilon_nStepSARSA = 0.1
  
  env = DeterministicGridWorld(sizeX, sizeY, startStates=startStates, terminalStates=terminalStates,
    impassableStates=impassableStates, defaultReward=defaultReward, specialRewards=specialRewards)

  env.printEnv()
  
  nStepsPerEpisode_DynaQ = []
  cum_rewards_DynaQ = []
  for idx_experiment in range(1, nExperiments+1):
  
    print("Experiment : ", idx_experiment)

    for nPlanningSteps in nPlanningSteps_list:
      agent_DynaQ = DynaQ(env.nStates, env.nActions, alpha_DynaQ, gamma_DynaQ, nPlanningSteps, epsilon=epsilon_DynaQ)
      print("running:", agent_DynaQ.getName())
      cum_reward, nStepsPerEpisode = runExperiment(nEpisodes, env, agent_DynaQ)
      nStepsPerEpisode_DynaQ.append(nStepsPerEpisode)
      cum_rewards_DynaQ.append(cum_reward)
    
    agent_nStepSARSA = nStepSARSA(env.nStates, env.nActions, alpha_nStepSARSA, gamma_nStepSARSA, n_nStepSARSA, epsilon=epsilon_nStepSARSA)
    print("running:", agent_nStepSARSA.getName())
    cum_reward_nStepSARSA, nStepsPerEpisode_nStepSARSA = runExperiment(nEpisodes, env, agent_nStepSARSA)
    
    agent_nStepTB = nStepTreeBackup(env.nStates, env.nActions, alpha_nStepTB, gamma_nStepTB, n_nStepTB)
    print("running:", agent_nStepTB.getName())
    policy_behaviour = StochasticPolicy(env.nStates, env.nActions, policyUpdateMethod="esoft", epsilon=epsilon_behaviourPolicy)
    cum_reward_nStepTB, nStepsPerEpisode_nStepTB = runExperiment(nEpisodes, env, agent_nStepTB, policy_behaviour, doUpdateBehaviourPolicy)

    agent_nStepQSigma = nStepQSigma(env.nStates, env.nActions, alpha_nStepQSigma, gamma_nStepQSigma, n_nStepQSigma, sigma_nStepQSigma)
    print("running:", agent_nStepQSigma.getName())
    policy_behaviour = StochasticPolicy(env.nStates, env.nActions, policyUpdateMethod="esoft", epsilon=epsilon_behaviourPolicy)
    cum_reward_nStepQSigma, nStepsPerEpisode_nStepQSigma = runExperiment(nEpisodes, env, agent_nStepQSigma, policy_behaviour, doUpdateBehaviourPolicy)
  
  pl.figure()
  for i, nPlanningSteps in enumerate(nPlanningSteps_list):
    pl.plot(nStepsPerEpisode_DynaQ[i], label='Dyna-Q '+str(nPlanningSteps)+' planning steps')
  pl.plot(nStepsPerEpisode_nStepSARSA, label=str(n_nStepSARSA)+' Step SARSA')
  pl.plot(nStepsPerEpisode_nStepTB, label=str(n_nStepTB)+' step Tree Backup')
  pl.plot(nStepsPerEpisode_nStepQSigma, label=str(n_nStepQSigma)+' step QSigma')
  pl.legend()
  pl.xlabel("Episodes")
  pl.ylabel("Steps per episode")
  pl.figure()
  for i, nPlanningSteps in enumerate(nPlanningSteps_list):
    pl.plot(cum_rewards_DynaQ[i], label='Dyna-Q '+str(nPlanningSteps)+' planning steps')
  pl.plot(cum_reward_nStepSARSA, label=str(n_nStepSARSA)+' step SARSA')
  pl.plot(cum_reward_nStepTB, label=str(n_nStepTB)+' step Tree Backup')
  pl.plot(cum_reward_nStepQSigma, label=str(n_nStepQSigma)+' step QSigma')
  pl.legend()
  pl.xlabel("Timesteps")
  pl.ylabel("Average cumulative reward")
  pl.show()
  
  agents = [agent_DynaQ, agent_nStepSARSA, agent_nStepTB, agent_nStepQSigma ]
  for agent in agents:
    print("Policy for :", agent.getName())
    env.printEnv(agent)
  
  for agent in agents:
    input("Press any key to simulate agent "+agent.getName())
    agentHistory = runSimulation(env, agent, 200)
    print("Simulation:", agent.getName())
    env.render(agentHistory)