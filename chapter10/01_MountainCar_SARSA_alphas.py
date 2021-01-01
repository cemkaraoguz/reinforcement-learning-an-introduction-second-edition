'''
01_MountainCar_SARSA_alphas.py : Replication of Figure 10.2

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from IRL.environments.Cars import MountainCar
from IRL.agents.TemporalDifferenceApproximation import SemiGradientSARSA
from IRL.utils.ApproximationFunctions import linearTransform, dLinearTransform
from IRL.utils.FeatureTransformations import tileCoding

def runExperiment(nEpisodes, env, agent):
  nStepsPerEpisode = np.zeros(nEpisodes)
  for e in range(nEpisodes):
    
    if e%10==0:
      print("Episode : ", e)
    
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
      action = new_action
      t += 1
      
    nStepsPerEpisode[e] = t
  
  return nStepsPerEpisode
  
if __name__=="__main__":

  nExperiments = 100
  nEpisodes = 500
  
  # Environment
  positionBounds = [-1.2, 0.5]
  velocityBounds = [-0.07, 0.07]
  startPositionBounds = [-0.6, -0.4]
  
  # Agent
  alphas = [0.1/8, 0.2/8, 0.5/8]
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

  env = MountainCar(positionBounds, velocityBounds, startPositionBounds)
  
  nStepsPerEpisode_avg_all = []
  for alpha in alphas:
    nStepsPerEpisode_avg = np.zeros(nEpisodes)
    for idx_experiment in range(1, nExperiments+1):
      
      print("alpha:", alpha, "idxExperiment", idx_experiment)
      
      agent = SemiGradientSARSA(nParams, nActions, alpha, gamma, approximationFunctionArgs=approximationFunctionArgs, epsilon=epsilon)
      nStepsPerEpisode = runExperiment(nEpisodes, env, agent)
      nStepsPerEpisode_avg = nStepsPerEpisode_avg + (1.0/idx_experiment)*(nStepsPerEpisode - nStepsPerEpisode_avg)
    nStepsPerEpisode_avg_all.append(nStepsPerEpisode_avg)
    
  pl.figure()
  for i, alpha in enumerate(alphas):
    pl.plot(nStepsPerEpisode_avg_all[i], label="alpha="+str(round(alpha,5)))
  pl.xlabel('Episodes')
  pl.ylabel('Number of steps per episode')
  pl.legend()
  pl.show()
  