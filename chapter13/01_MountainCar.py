'''
01_MountainCar.py : Application of Actor-Critic methods to Mountain Car problem

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from IRL.environments.Cars import MountainCar
from IRL.agents.PolicyGradient import OneStepActorCritic, ActorCriticWithEligibilityTraces
from IRL.utils.ApproximationFunctions import linearTransform, dLinearTransform, softmaxLinear, dLogSoftmaxLinear
from IRL.utils.FeatureTransformations import tileCoding
from IRL.utils.Helpers import runSimulation

def showCostToGo(agent, episode, nStates=20, doShowNow=False):
  state_pos = np.linspace(positionBounds[0], positionBounds[1],num=nStates)
  state_vel = np.linspace(velocityBounds[0], velocityBounds[1],num=nStates)
  cost_to_go = np.array([[-np.max(agent.getValue([p,v])) for p in state_pos] for v in state_vel])
  fig = pl.figure()
  ax = fig.add_subplot(111, projection='3d')
  (X, Y), Z = np.meshgrid(state_pos, state_vel), cost_to_go
  ax.set_xlabel('Position', fontsize=10)
  ax.set_ylabel('Velocity', fontsize=10)
  ax.set_zlabel('Cost-to-go', fontsize=10)
  ax.set_title(agent.getName()+" Cost-to-go function in Episode "+str(episode))
  ax.plot_surface(X, Y, Z)
  if doShowNow: pl.show()

def runExperiment(nEpisodes, env, agent, episodesToShowCostToGo, nStatesVis=50, doShowNow=False):
  nStepsPerEpisode = np.zeros(nEpisodes)
  for e in range(nEpisodes):
    
    if e%1==0:
      print("Episode : ", e)

    if e in episodesToShowCostToGo: showCostToGo(agent, e, nStatesVis, doShowNow)
    
    experiences = [{}]
    done = False
    state = env.reset()
    action = agent.selectAction(state)
    t = 0

    while not done:     

      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done

      new_state, reward, done = env.step(action)
      
      #print("Episode:", e, "State:", state, "Action: ", env.actionMapping[action][1], "Reward: ", reward, "New state:", new_state, "done:", done)

      new_action = agent.selectAction(new_state)
      
      xp = {}
      xp['reward'] = reward
      xp['state'] = new_state
      xp['done'] = done
      xp['action'] = new_action
      experiences.append(xp)

      agent.update(experiences)
      
      state = new_state
      if("Expected" in agent.getName()):
        action = agent.selectAction(new_state)
      else:
        action = new_action
      t += 1
      
    nStepsPerEpisode[e] = t
  
  return nStepsPerEpisode
  
if __name__=="__main__":

  nEpisodes = 200
  episodesToShowCostToGo = [1, 10, 100, 199]
  
  # Environment
  positionBounds = [-1.2, 0.5]
  velocityBounds = [-0.07, 0.07]
  startPositionBounds = [-0.6, -0.4]
  env = MountainCar(positionBounds, velocityBounds, startPositionBounds)
  
  # Agents
  alpha_w = 0.5/8
  alpha_theta = 0.05
  gamma = 1.0
  lambd_w = 0.8
  lambd_theta = 0.8
  
  nActions = 3
  minStates = [positionBounds[0], velocityBounds[0]]
  maxStates = [positionBounds[1], velocityBounds[1]]
  nTilings = 8
  tilingOffsets = [[i, j] for i, j in zip(np.linspace(-0.4, 0.4,num=nTilings), np.linspace(-0.04, 0.04,num=nTilings))] # (idxTiling, dimState)
  tilingSize = [[8, 8] for _ in range(nTilings)]  # (idxTiling, dimState)
  nParams_w = np.sum([np.prod(i) for i in tilingSize])
  approximationFunctionArgs = {'af':linearTransform, 'afd':dLinearTransform, 'ftf':tileCoding,
    'minStates':minStates, 'maxStates':maxStates, 'nTilings':nTilings, 
    'tilingOffsets':tilingOffsets, 'tilingSize':tilingSize}

  nTilings_theta = 1
  tilingOffsets_theta = [[0, 0]] # (idxTiling, dimState)
  tilingSize_theta = [[8, 8] for _ in range(nTilings_theta)]  # (idxTiling, dimState) 
  nParams_theta = env.nActions * np.sum([np.prod(i) for i in tilingSize_theta])
  policyApproximationFunctionArgs = {'af':softmaxLinear, 'afd':dLogSoftmaxLinear, 'ftf':tileCoding,
    'minStates':minStates, 'maxStates':maxStates, 'nTilings':nTilings_theta, 
    'tilingOffsets':tilingOffsets_theta, 'tilingSize':tilingSize_theta, 'nActions':env.nActions}

  agent_OSAC = OneStepActorCritic(alpha_w, alpha_theta, gamma, nParams_w, approximationFunctionArgs, 
    nParams_theta, env.nActions, policyApproximationFunctionArgs)
  nStepsPerEpisode_OSAC = runExperiment(nEpisodes, env, agent_OSAC, episodesToShowCostToGo, doShowNow=False)
  
  agent_ACET = ActorCriticWithEligibilityTraces(alpha_w, alpha_theta, gamma, lambd_w, lambd_theta, nParams_w, approximationFunctionArgs,
    nParams_theta, env.nActions, policyApproximationFunctionArgs)
  nStepsPerEpisode_ACET = runExperiment(nEpisodes, env, agent_ACET, episodesToShowCostToGo, doShowNow=False)
  
  pl.figure()
  pl.plot(nStepsPerEpisode_OSAC, label=agent_OSAC.getName())
  pl.plot(nStepsPerEpisode_ACET, label=agent_ACET.getName())
  pl.xlabel('Episodes')
  pl.ylabel('Number of steps per episode')
  pl.legend()
  pl.show()
  
  for agent in [agent_OSAC, agent_ACET]:
    input("Press any key to simulate agent "+agent.getName())
    agentHistory = runSimulation(env, agent, 500)