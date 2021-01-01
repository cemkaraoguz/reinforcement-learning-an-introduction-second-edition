'''
02_CangingMaze.py : Replication of Figure 8.4 and 8.5

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.TemporalDifferenceLearning import DynaQ
from IRL.utils.Policies import selectAction_egreedy

def runExperiment(nEpisodes, nTimesteps, env, agent, envChangeTimestep, doASB=False, kappa=0.0):
  nStepsPerEpisode = np.zeros(nEpisodes)
  timestep = 0
  e = 0
  cumulative_reward = [0]
  while timestep<nTimesteps:
    
    if(e%10==0):
      print("Episode : ", e)
      
    state = env.reset()
    if(doASB):
      boosted_qTable = agent.actionValueTable[state,:] + kappa*np.sqrt(agent.model.transitionBookkeeper[state,:])
      action = selectAction_egreedy(boosted_qTable, epsilon=agent.policy.epsilon)
    else:
      action = agent.selectAction(state)
    done = False
    experiences = [{}]
    while not done:
      
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
      
      timestep += 1
      if timestep==envChangeTimestep:
        state = changeEnvironment(env)
      if timestep>=nTimesteps:
        break
      cumulative_reward.append(cumulative_reward[-1]+reward)
      if e<nEpisodes:
        nStepsPerEpisode[e] += 1
        
    e+=1
  
  return cumulative_reward, nStepsPerEpisode
  
def changeEnvironment(env):
  env.setImpassableStates(newImpassableStates)
  env.generateModels()
  state = env.reset()
  return state
  
if __name__=="__main__":

  nExperiments = 100
  mazeType = "Shortcut"
  
  # Environment
  sizeX = 9
  sizeY = 6
  defaultReward = 0.0
  startStates = [(3,5)]
  terminalStates = [(8,0)]
  specialRewards = {((terminalStates[0][0],terminalStates[0][1]+1),0):1.0, 
    ((terminalStates[0][0]-1,terminalStates[0][1]),2):1.0}
  impassableStates = [(x,3) for x in range(0,8)]

  if(mazeType=="Blocking"):
    # Example 8.2
    nEpisodes = 100 
    nTimesteps = 3000
    envChangeTimestep = 1000
    impassableStates = [(x,3) for x in range(0,8)]
    newImpassableStates = [(x,3) for x in range(1,9)]
  elif(mazeType=="Shortcut"):
    # Example 8.3 
    nEpisodes = 250 
    nTimesteps = 6000
    envChangeTimestep = 3000
    impassableStates = [(x,3) for x in range(1,9)]
    newImpassableStates = [(x,3) for x in range(1,8)]
  else:
    nEpisodes = 100 
    nTimesteps = 3000
    envChangeTimestep = 0
    impassableStates = [(x,3) for x in range(0,8)]
    newImpassableStates = [(x,3) for x in range(0,8)]
    
  # Agent DynaQ
  alpha_DynaQ = 0.9
  gamma_DynaQ = 0.95
  epsilon_DynaQ = 0.1
  nPlanningSteps_DynaQ = 5
  kappa_DynaQ = 0.0
  
  # Agent DynaQPlus
  alpha_DynaQPlus = 0.9
  gamma_DynaQPlus = 0.95
  epsilon_DynaQPlus = 0.0
  nPlanningSteps_DynaQPlus = 5
  kappa_DynaQPlus = 0.001
  
  # Agent DynaQ ASB
  alpha_DynaQ_ASB = 0.9
  gamma_DynaQ_ASB = 0.95
  epsilon_DynaQ_ASB = 0.2
  nPlanningSteps_DynaQ_ASB = 5
  kappa_ASB = 0.001
  
  avg_cum_reward_DynaQ = np.zeros(nTimesteps)
  avg_cum_reward_DynaQPlus = np.zeros(nTimesteps)
  avg_cum_reward_DynaQ_ASB = np.zeros(nTimesteps)
  for idxExperiment in range(1, nExperiments+1):
    
    print("Experiment:", idxExperiment)
    
    env = DeterministicGridWorld(sizeX, sizeY, startStates=startStates, terminalStates=terminalStates,
      impassableStates=impassableStates, defaultReward=defaultReward, specialRewards=specialRewards)
    agent_DynaQ = DynaQ(env.nStates, env.nActions, alpha_DynaQ, gamma_DynaQ, nPlanningSteps_DynaQ, kappa=kappa_DynaQ, epsilon=epsilon_DynaQ)
    cumulative_reward_DynaQ, nStepsPerEpisode_DynaQ = runExperiment(nEpisodes, nTimesteps, env, agent_DynaQ, envChangeTimestep)
    avg_cum_reward_DynaQ +=  (1.0/idxExperiment)*(cumulative_reward_DynaQ - avg_cum_reward_DynaQ)
    
    env = DeterministicGridWorld(sizeX, sizeY, startStates=startStates, terminalStates=terminalStates,
      impassableStates=impassableStates, defaultReward=defaultReward, specialRewards=specialRewards)
    agent_DynaQPlus = DynaQ(env.nStates, env.nActions, alpha_DynaQPlus, gamma_DynaQPlus, nPlanningSteps_DynaQPlus, 
      kappa=kappa_DynaQPlus, epsilon=epsilon_DynaQPlus)
    cumulative_reward_DynaQPlus, nStepsPerEpisode_DynaQPlus = runExperiment(nEpisodes, nTimesteps, env, agent_DynaQPlus, envChangeTimestep)   
    avg_cum_reward_DynaQPlus +=  (1.0/idxExperiment)*(cumulative_reward_DynaQPlus - avg_cum_reward_DynaQPlus)

    env = DeterministicGridWorld(sizeX, sizeY, startStates=startStates, terminalStates=terminalStates,
      impassableStates=impassableStates, defaultReward=defaultReward, specialRewards=specialRewards)
    agent_DynaQ_ASB = DynaQ(env.nStates, env.nActions, alpha_DynaQ_ASB, gamma_DynaQ_ASB, nPlanningSteps_DynaQ_ASB, epsilon=epsilon_DynaQ_ASB)
    cumulative_reward_DynaQ_ASB, nStepsPerEpisode_DynaQ_ASB = runExperiment(nEpisodes, nTimesteps, env, agent_DynaQ_ASB, 
      envChangeTimestep, True, kappa_ASB)
    avg_cum_reward_DynaQ_ASB +=  (1.0/idxExperiment)*(cumulative_reward_DynaQ_ASB - avg_cum_reward_DynaQ_ASB)
  
  pl.figure()
  pl.plot(avg_cum_reward_DynaQ, label='Dyna-Q')
  pl.plot(avg_cum_reward_DynaQPlus, label='Dyna-Q+')
  pl.plot(avg_cum_reward_DynaQ_ASB, label='Dyna-Q ASB')
  pl.legend()
  pl.xlabel("Time steps")
  pl.ylabel("Cumulative reward")
  pl.show()
  
  agents = [agent_DynaQ, agent_DynaQPlus]
  for agent in agents:
    print("Policy for :", agent.getName())
    env.printEnv(agent)
