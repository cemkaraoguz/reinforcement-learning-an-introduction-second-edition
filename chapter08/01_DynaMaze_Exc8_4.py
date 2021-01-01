'''
01_DynaMaze_Exc8_4.py : Solution to Exercise 8.4

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.TemporalDifferenceLearning import DynaQ
from IRL.utils.Policies import selectAction_egreedy

def runExperiment(nTimesteps, env, agent, doASB=False, kappa=0.0):
  nStepsPerEpisode = []
  cumulative_reward = [0]
  t = 0
  while t<nTimesteps:
    state = env.reset()
    if(doASB):
      boosted_qTable = agent.actionValueTable[state,:] + kappa*np.sqrt(agent.model.transitionBookkeeper[state,:])
      action = selectAction_egreedy(boosted_qTable, epsilon=agent.policy.epsilon)
    else:
      action = agent.selectAction(state)
    done = False
    experiences = [{}]
    nStepsPerEpisode.append(0)
    while not done:
    
      if(t%1000==0):
        print("t : ", t)

      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done
      
      new_state, reward, done = env.step(action)
      
      #print("Episode:", e, "State:", state, "Action: ", env.actionMapping[action][1], "Reward: ", reward, "New state:", new_state, "done:", done)
      
      if(doASB):
        boosted_qTable = agent.actionValueTable[new_state,:] + kappa*np.sqrt(agent.model.transitionBookkeeper[new_state,:])
        new_action = selectAction_egreedy(boosted_qTable, epsilon=agent.policy.epsilon)
      else:
        new_action = agent.selectAction(new_state)
      
      xp = {}
      xp['state'] = new_state
      xp['action'] = new_action
      xp['reward'] = reward
      xp['done'] = done
      experiences.append(xp)
      
      agent.update(experiences[-2:])
      
      state = new_state
      action = new_action
      
      cumulative_reward.append(cumulative_reward[-1]+reward)
      nStepsPerEpisode[-1] += 1
      t += 1
      
  return np.array(cumulative_reward[0:nTimesteps]), nStepsPerEpisode
  
if __name__=="__main__":

  nExperiments = 5
  nTimesteps = 3000 

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
  alpha_DynaQ = 0.5
  gamma_DynaQ = 0.95
  epsilon_DynaQ = 0.1
  nPlanningSteps_list = [50]

  alpha_DynaQPlus = 0.5
  gamma_DynaQPlus = 0.95
  epsilon_DynaQPlus = 0.0
  nPlanningSteps_DynaQPlus = 50
  kappa_DynaQPlus = 0.001
  
  alpha_DynaQ_ASB = 0.5
  gamma_DynaQ_ASB = 0.95
  epsilon_DynaQ_ASB = 0.0
  nPlanningSteps_ASB = 50
  kappa_ASB = 0.001
  
  env = DeterministicGridWorld(sizeX, sizeY, startStates=startStates, terminalStates=terminalStates,
    impassableStates=impassableStates, defaultReward=defaultReward, specialRewards=specialRewards)

  env.printEnv()
  
  nStepsPerEpisode_list = []
  avgCumReward_DynaQ = []
  for i in range(len(nPlanningSteps_list)):
    avgCumReward_DynaQ.append(np.zeros(nTimesteps))
  avgCumReward_DynaQPlus = np.zeros(nTimesteps)
  avgCumReward_DynaQ_ASB = np.zeros(nTimesteps)
  for idx_experiment in range(1, nExperiments+1):
  
    print("Experiment : ", idx_experiment)

    for i, nPlanningSteps in enumerate(nPlanningSteps_list):
      agent_DynaQ = DynaQ(env.nStates, env.nActions, alpha_DynaQ, gamma_DynaQ, nPlanningSteps, epsilon=epsilon_DynaQ)
      cumulative_reward_DynaQ, nStepsPerEpisode = runExperiment(nTimesteps, env, agent_DynaQ)
      nStepsPerEpisode_list.append(nStepsPerEpisode)      
      avgCumReward_DynaQ[i] = avgCumReward_DynaQ[i] + (1.0/idx_experiment)*(cumulative_reward_DynaQ - avgCumReward_DynaQ[i])

    agent_DynaQPlus = DynaQ(env.nStates, env.nActions, alpha_DynaQPlus, gamma_DynaQPlus, 
      nPlanningSteps_DynaQPlus, kappa=kappa_DynaQPlus, epsilon=epsilon_DynaQPlus)
    cumulative_reward_DynaQPlus, nStepsPerEpisode_DynaQPlus = runExperiment(nTimesteps, env, agent_DynaQPlus)
    avgCumReward_DynaQPlus = avgCumReward_DynaQPlus + (1.0/idx_experiment)*(cumulative_reward_DynaQPlus - avgCumReward_DynaQPlus)
  
    agent_DynaQ_ASB = DynaQ(env.nStates, env.nActions, alpha_DynaQ_ASB, gamma_DynaQ_ASB, nPlanningSteps_ASB, epsilon=epsilon_DynaQ_ASB)
    cumulative_reward_DynaQ_ASB, nStepsPerEpisode_ASB = runExperiment(nTimesteps, env, agent_DynaQ_ASB, doASB=True, kappa=kappa_ASB)
    avgCumReward_DynaQ_ASB = avgCumReward_DynaQ_ASB + (1.0/idx_experiment)*(cumulative_reward_DynaQ_ASB - avgCumReward_DynaQ_ASB)
  
  for i, nPlanningSteps in enumerate(nPlanningSteps_list):
    print(nPlanningSteps, " steps Dyna-Q std:", np.std(nStepsPerEpisode_list[i]))
  print(nPlanningSteps_DynaQPlus, " steps Dyna-Q+ std:", np.std(nStepsPerEpisode_DynaQPlus))
  print(nPlanningSteps_ASB, " steps Dyna-Q ASB std:", np.std(nStepsPerEpisode_ASB))
  
  pl.figure()
  for i, nPlanningSteps in enumerate(nPlanningSteps_list):
    pl.plot(nStepsPerEpisode_list[i], label='Dyna-Q '+str(nPlanningSteps)+' planning steps')
  pl.plot(nStepsPerEpisode_DynaQPlus, label='Dyna-Q+ '+str(nPlanningSteps_DynaQPlus)+' planning steps')
  pl.plot(nStepsPerEpisode_ASB, label='Dyna-Q ASB '+str(nPlanningSteps_ASB)+' planning steps')
  pl.legend()
  pl.xlabel("Episodes")
  pl.ylabel("Steps per episode")
  pl.figure()
  for i, nPlanningSteps in enumerate(nPlanningSteps_list):
    pl.plot(avgCumReward_DynaQ[i], label='Dyna-Q '+str(nPlanningSteps)+' planning steps')
  pl.plot(avgCumReward_DynaQPlus, label='Dyna-Q+ '+str(nPlanningSteps_DynaQPlus)+' planning steps')
  pl.plot(avgCumReward_DynaQ_ASB, label='Dyna-Q ASB '+str(nPlanningSteps_ASB)+' planning steps')
  pl.legend()
  pl.xlabel("Time steps")
  pl.ylabel("Average cumulative reward")
  pl.show()
  
  agents = [agent_DynaQ, agent_DynaQPlus, agent_DynaQ_ASB]
  for agent in agents:
    print("Policy for :", agent.getName())
    env.printEnv(agent)
