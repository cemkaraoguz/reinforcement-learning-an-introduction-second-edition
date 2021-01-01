'''
03_MountainCar_nStepSARSA_alphas.py : Replication of Figure 10.4

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from IRL.environments.Cars import MountainCar
from IRL.agents.TemporalDifferenceApproximation import nStepSemiGradientSARSA
from IRL.utils.ApproximationFunctions import linearTransform, dLinearTransform
from IRL.utils.FeatureTransformations import tileCoding

def runExperiment(nEpisodes, env, agent, maxStepsPerEpisode=None):
  nStepsPerEpisode = np.zeros(nEpisodes)
  for e in range(nEpisodes):
    
    if e%1==0:
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
      
      if maxStepsPerEpisode is not None and t>=maxStepsPerEpisode:
        break
      
    nStepsPerEpisode[e] = t
  
  return nStepsPerEpisode
  
if __name__=="__main__":

  nExperiments = 100
  nEpisodes = 50
  maxStepsPerEpisode = 1000
  
  # Environment
  positionBounds = [-1.2, 0.5]
  velocityBounds = [-0.07, 0.07]
  startPositionBounds = [-0.6, -0.4]
  
  # Agent
  alphas = [0.2/8, 0.4/8, 0.6/8, 0.8/8, 1.0/8, 1.2/8, 1.4/8]
  gamma = 1.0
  epsilon = 0.0
  nValues = [1, 2, 4, 8, 16]
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
  for n in nValues:
    for alpha in alphas:
      nStepsPerEpisode_avg = 0.0
      for idx_experiment in range(1, nExperiments+1):
        
        print("n", n, "alpha:", alpha, "idxExperiment", idx_experiment)
        
        agent = nStepSemiGradientSARSA(nParams, nActions, alpha, gamma, n, approximationFunctionArgs=approximationFunctionArgs, epsilon=epsilon)
        nStepsPerEpisode = runExperiment(nEpisodes, env, agent, maxStepsPerEpisode=maxStepsPerEpisode)
        nStepsPerEpisode_avg = nStepsPerEpisode_avg + (1.0/idx_experiment)*(np.mean(nStepsPerEpisode) - nStepsPerEpisode_avg)
      nStepsPerEpisode_avg_all.append(nStepsPerEpisode_avg)
  
  fig, ax = pl.subplots()
  for i, n in enumerate(nValues):
    plotarray = nStepsPerEpisode_avg_all[i*len(alphas):i*len(alphas)+len(alphas)]
    ax.plot(plotarray, label=str(n)+" step TD")
  ax.set_xlabel("alpha")
  ax.set_ylabel("Number of steps per episode averaged over "+str(nExperiments)+" experiments")
  ax.set_xticks(range(len(alphas)))
  ax.set_xticklabels([str(np.round(i,4)) for i in alphas])
  pl.legend()
  pl.show()
  