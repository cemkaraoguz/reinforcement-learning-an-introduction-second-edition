'''
03_DynaMaze_PrioritizedSweeping.py : Replication of figure in Example 8.4

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.TemporalDifferenceLearning import DynaQ, PrioritizedSweeping
from IRL.utils.Helpers import runSimulation

def runExperiment(nTimesteps, env, agent, doASB=False, kappa=0.0, lenPolicyStableHistory=10, rateStateChange=0.01):
  nStepsPerEpisode = []
  cumulative_reward = [0]
  t = 0
  e = 0
  nUpdatesExperiment = 0
  policyStableHistory = np.zeros(lenPolicyStableHistory)+env.nStates
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
    maxTDError_episode = 0.0
    rewards = []
    policy_before = np.zeros(env.nStates)
    for s in range(env.nStates):
      policy_before[s] = agent.getGreedyAction(s)
    while not done:
    
      if(t%1000==0):
        print(agent.getName(), "Episode:", len(nStepsPerEpisode), "t:", t, "nStepsPerEpisode:", nStepsPerEpisode[-1], "max TDError:", maxTDError_episode)
        print("policy stability history", policyStableHistory)
        
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
      
      maxTDError, nUpdates = agent.update(experiences[-2:])
      maxTDError_episode = max(maxTDError_episode, maxTDError)
      nUpdatesExperiment += nUpdates
      
      state = new_state
      action = new_action
      
      cumulative_reward.append(cumulative_reward[-1]+reward)
      nStepsPerEpisode[-1] += 1
      t += 1
    
    policy_after = np.zeros(env.nStates)
    for s in range(env.nStates):
      policy_after[s] = agent.getGreedyAction(s)
    policyStableHistory[e%len(policyStableHistory)] = np.sum(policy_after!=policy_before)
    e+=1
    if policyStableHistory.mean()<(rateStateChange*env.nStates):
      break
      
  return np.array(cumulative_reward[0:nTimesteps]), nStepsPerEpisode, nUpdatesExperiment
  
def scaleEnv(scaleFactor, sizeX, sizeY, startStates, terminalStates, impassableStates_bounds):
  
  assert scaleFactor>0
  
  sizeX_scaled = int(sizeX * scaleFactor)
  sizeY_scaled = int(sizeY * scaleFactor)
  startStates_scaled = [(int(s[0]*scaleFactor), int(s[1]*scaleFactor)) for s in startStates]
  terminalStates_scaled = [(int(s[0]*scaleFactor), int(s[1]*scaleFactor)) for s in terminalStates]  
  impassableStates_bounds_scaled = [(int(b[0]*scaleFactor), int(b[1]*scaleFactor), int(b[2]*scaleFactor), int(b[3]*scaleFactor)) for b in impassableStates_bounds]
  impassableStates_scaled = computeImpassableStates(impassableStates_bounds_scaled)
    
  print("scaleFactor", scaleFactor)
  print("input", sizeX, sizeY, startStates, terminalStates)
  print("scaled", sizeX_scaled, sizeY_scaled, startStates_scaled, terminalStates_scaled)
  
  return sizeX_scaled, sizeY_scaled, startStates_scaled, terminalStates_scaled, impassableStates_scaled
  
def computeImpassableStates(impassableStates_bounds):
  impassableStates = []
  for bound in impassableStates_bounds:
    for x in range(bound[0],bound[2]):
      for y in range(bound[1],bound[3]):
        impassableStates.extend([(x,y)])
  return impassableStates
  
if __name__=="__main__":

  nExperiments = 1
  nTimesteps = 300000 
  doCheckInterimResults = True

  # Environment
  sizeX = 9
  sizeY = 6
  defaultReward = 0.0
  startStates = [(0,2)]
  terminalStates = [(8,0)]
  specialRewards = {((terminalStates[0][0],terminalStates[0][1]+1),0):1.0, 
    ((terminalStates[0][0]-1,terminalStates[0][1]),2):1.0}
  impassableStates_bounds = [(5,4,6,5),(2,1,3,4),(7,0,8,3)]
  impassableStates = computeImpassableStates(impassableStates_bounds)
  env_scales = [1,2,3,4,5,6]

  # Agent
  alpha_DynaQ = 0.5
  gamma_DynaQ = 0.95
  epsilon_DynaQ = 0.0
  nPlanningSteps_DynaQ = 5

  alpha_PS = 0.5
  gamma_PS = 0.95
  epsilon_PS = 0.0
  nPlanningSteps_PS = 5
  theta_PS = 1e-5
  
  # Optimality
  lenPolicyStableHistory = 10
  rateStateChange = 0.01
  
  nUpdates_list_PS = []
  nUpdates_list_DynaQ = []
  for scale in env_scales:
  
    print("Scale : ", scale)

    sizeX_scaled, sizeY_scaled, startStates_scaled, terminalStates_scaled, impassableStates_scaled = scaleEnv(scale, sizeX, sizeY, startStates, terminalStates, impassableStates_bounds)
    specialRewards_scaled = {((terminalStates_scaled[0][0],terminalStates_scaled[0][1]+1),0):1.0, 
      ((terminalStates_scaled[0][0]-1,terminalStates_scaled[0][1]),2):1.0}
    env = DeterministicGridWorld(sizeX_scaled, sizeY_scaled, startStates=startStates_scaled, terminalStates=terminalStates_scaled,
      impassableStates=impassableStates_scaled, defaultReward=defaultReward, specialRewards=specialRewards_scaled)
    
    env.printEnv()
    
    agent_PS = PrioritizedSweeping(env.nStates, env.nActions, alpha_PS, gamma_PS, nPlanningSteps_PS, theta=theta_PS, epsilon=epsilon_PS)
    cumulative_reward_PS, nStepsPerEpisode_PS, nUpdates_PS = runExperiment(nTimesteps, env, agent_PS, 
      lenPolicyStableHistory=lenPolicyStableHistory, rateStateChange=rateStateChange)
    agent_DynaQ = DynaQ(env.nStates, env.nActions, alpha_DynaQ, gamma_DynaQ, nPlanningSteps_DynaQ, epsilon=epsilon_DynaQ)
    cumulative_reward_DynaQ, nStepsPerEpisode_DynaQ, nUpdates_DynaQ = runExperiment(nTimesteps, env, agent_DynaQ, 
      lenPolicyStableHistory=lenPolicyStableHistory, rateStateChange=rateStateChange)

    print(nPlanningSteps_DynaQ, " steps DynaQ updates until convergence:", nUpdates_DynaQ)
    print(nPlanningSteps_PS, " steps Prioritized Sweeping updates until convergence:", nUpdates_PS)
    
    nUpdates_list_PS.append(nUpdates_PS)
    nUpdates_list_DynaQ.append(nUpdates_DynaQ)
    
    if(doCheckInterimResults):
      pl.figure()
      pl.plot(nStepsPerEpisode_DynaQ, label='Dyna-Q '+str(nPlanningSteps_DynaQ)+' planning steps')
      pl.plot(nStepsPerEpisode_PS, label='Prioritized Sweeping '+str(nPlanningSteps_PS)+' planning steps')
      pl.legend()
      pl.xlabel("Episodes")
      pl.ylabel("Steps per episode")
      pl.figure()
      pl.plot(cumulative_reward_DynaQ, label='Dyna-Q '+str(nPlanningSteps_DynaQ)+' planning steps')
      pl.plot(cumulative_reward_PS, label='Prioritized Sweeping '+str(nPlanningSteps_PS)+' planning steps')
      pl.legend()
      pl.xlabel("Time steps")
      pl.ylabel("Cumulative reward")
      pl.show()
    
      agents = [agent_DynaQ, agent_PS]
      for agent in agents:
        print("Policy for :", agent.getName())
        env.printEnv(agent)
      
      for agent in agents:
        input("Press any key to simulate agent "+agent.getName())
        agentHistory = runSimulation(env, agent, 200)
        print("Simulation:", agent.getName()) 
        env.render(agentHistory)
  
  nstatesgw = [i*i*sizeX*sizeY for i in env_scales]
  pl.figure()
  pl.plot(nstatesgw, nUpdates_list_DynaQ, label='Dyna-Q '+str(nPlanningSteps_DynaQ)+' planning steps')
  pl.plot(nstatesgw, nUpdates_list_PS, label='Prioritized Sweeping '+str(nPlanningSteps_PS)+' planning steps')
  pl.legend()
  pl.xlabel("Gridworld Size (# of states)")
  pl.ylabel("Updates until optimal solution")
  pl.show()
    
  