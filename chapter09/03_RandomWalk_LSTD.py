'''
03_RandomWalk_LSTD.py : Application of Least Squares TD algorithm on a simple Random Walk task

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.ToyExamples import RandomWalk
from IRL.agents.TemporalDifferenceApproximation import LeastSquaresTD
from IRL.utils.FeatureTransformations import stateAggregation, radialBasisFunction
from IRL.utils.ApproximationFunctions import linearTransform, dLinearTransform

if __name__=="__main__":  
  nEpisodes = 2000
  
  # Environment
  nStatesOneSide = 50
  specialRewards = {nStatesOneSide*2:1.0, 0:-1.0}
  groundTruth = np.zeros(nStatesOneSide*2+1)
  groundTruth[nStatesOneSide:] = np.arange(nStatesOneSide+1)/nStatesOneSide
  groundTruth[0:nStatesOneSide] = np.arange(nStatesOneSide,0,-1)/(-nStatesOneSide)
  groundTruth = groundTruth[1:nStatesOneSide*2]
  nStates = nStatesOneSide*2+1
  
  # Agents
  epsilon_LSTD = 5e-4
  gamma_LSTD = 1.0
  nParams_LSTD = 20
  approximationFunctionArgs_LSTD_sa = {'af':linearTransform, 'afd':dLinearTransform, 
    'ftf':stateAggregation, 'nStates':nStates, 'nParams':nParams_LSTD}

  mu_rbf = np.linspace(0, nStates,num=nParams_LSTD+2)[1:-1]
  mu_rbf = np.linspace(0, nStates,num=nParams_LSTD)
  mu_rbf = np.linspace(-10, nStates+10,num=nParams_LSTD)
  sigma_rbf = np.ones(nParams_LSTD)*10.0
  approximationFunctionArgs_LSTD_rbf = {'af':linearTransform, 'afd':dLinearTransform, 
    'ftf':radialBasisFunction, 'mu':mu_rbf, 'sigma':sigma_rbf}
                    
  env = RandomWalk(nStatesOneSide, specialRewards=specialRewards)

  agent_LSTD = LeastSquaresTD(nParams_LSTD, gamma_LSTD, epsilon_LSTD, approximationFunctionArgs=approximationFunctionArgs_LSTD_rbf)

  for e in range(nEpisodes):
    
    if e%100==0:
      print("Episode : ", e)

    experiences = [{}]
    done = False
    state = env.reset()

    while not done:     
      experiences[-1]['state'] = state
      experiences[-1]['done'] = done

      new_state, reward, done = env.step()
      
      experiences[-1]['action'] = env.lastAction

      xp = {}
      xp['reward'] = reward
      xp['state'] = new_state
      xp['done'] = done
      experiences.append(xp)

      agent_LSTD.update(experiences)
      
      state = new_state
  
  estimatedValues_LSTD = [agent_LSTD.getValue(state) for state in range(env.nStates-1)]

  pl.figure()
  pl.plot(groundTruth, 'k', label="Real values")
  pl.plot(estimatedValues_LSTD, label=agent_LSTD.getName())
  pl.xlabel("State")
  pl.ylabel("Value")
  pl.legend()
  pl.show()
  