'''
00_MountainCar_SARSA.py : Replication of Figure 10.1

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from IRL.environments.Cars import MountainCar
from IRL.agents.TemporalDifferenceApproximation import SemiGradientSARSA, SemiGradientExpectedSARSA
from IRL.utils.ApproximationFunctions import linearTransform, dLinearTransform
from IRL.utils.FeatureTransformations import tileCoding

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
  ax.set_title("Cost-to-go function in Episode "+str(episode))
  ax.plot_surface(X, Y, Z)
  if doShowNow: pl.show()

def runExperiment(nEpisodes, env, agent, episodesToShowCostToGo, nStatesVis=50, doShowNow=False):
  nStepsPerEpisode = np.zeros(nEpisodes)
  for e in range(nEpisodes):
    
    if e%10==0:
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

  nEpisodes = 10000
  episodesToShowCostToGo = [1, 12, 104, 1000, 9000]
  
  # Environment
  positionBounds = [-1.2, 0.5]
  velocityBounds = [-0.07, 0.07]
  startPositionBounds = [-0.6, -0.4]
  
  # Agent
  alpha = 0.5/8
  gamma = 1.0
  epsilon = 0.0
  nActions = 3
  minStates = [positionBounds[0], velocityBounds[0]]
  maxStates = [positionBounds[1], velocityBounds[1]]
  nTilings = 8
  tilingOffsets = [[i, j] for i, j in zip(np.linspace(-0.4, 0.4,num=nTilings), np.linspace(-0.04, 0.04,num=nTilings))] # (idxTiling, dimState)
  tilingSize = [[8, 8] for _ in range(nTilings)]  # (idxTiling, dimState)
  nParams = nActions * np.sum([np.prod(i) for i in tilingSize])
  approximationFunctionArgs = {'af':linearTransform, 'afd':dLinearTransform, 'ftf':tileCoding,
    'minStates':minStates, 'maxStates':maxStates, 'nTilings':nTilings, 
    'tilingOffsets':tilingOffsets, 'tilingSize':tilingSize, 'nActions':nActions}

  alpha_expectedSARSA = 0.5/8
  gamma_expectedSARSA = 1.0
  epsilon_expectedSARSA = 0.0
  
  env = MountainCar(positionBounds, velocityBounds, startPositionBounds)
  agent = SemiGradientSARSA(nParams, nActions, alpha, gamma, approximationFunctionArgs=approximationFunctionArgs, epsilon=epsilon)
  # Expected SARSA agent for comparison
  agent_expectedSARSA = SemiGradientExpectedSARSA(nParams, nActions, alpha_expectedSARSA, gamma_expectedSARSA, approximationFunctionArgs=approximationFunctionArgs, epsilon=epsilon)
  
  nStepsPerEpisode = runExperiment(nEpisodes, env, agent, episodesToShowCostToGo, doShowNow=False)
  nStepsPerEpisode_expectedSARSA = runExperiment(nEpisodes, env, agent_expectedSARSA, episodesToShowCostToGo, doShowNow=False)
  
  pl.figure()
  pl.plot(nStepsPerEpisode, label=agent.getName())
  pl.plot(nStepsPerEpisode_expectedSARSA, label=agent_expectedSARSA.getName())
  pl.xlabel('Episodes')
  pl.ylabel('Number of steps per episode')
  pl.legend()
  pl.show()
  