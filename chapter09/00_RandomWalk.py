'''
00_RandomWalk.py : Replication of figure 9.1

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.ToyExamples import RandomWalk
from IRL.agents.MonteCarloApproximation import GradientMonteCarloPrediction
from IRL.utils.FeatureTransformations import stateAggregation
from IRL.utils.ApproximationFunctions import linearTransform, dLinearTransform

if __name__=="__main__":  
  nEpisodes = 10000
  
  # Environment
  nStatesOneSide = 250
  specialRewards = {nStatesOneSide*2:1.0, 0:-1.0}
  groundTruth = np.zeros(nStatesOneSide*2+1)
  groundTruth[nStatesOneSide:] = np.arange(nStatesOneSide+1)/nStatesOneSide
  groundTruth[0:nStatesOneSide] = np.arange(nStatesOneSide,0,-1)/(-nStatesOneSide)
  groundTruth = groundTruth[1:nStatesOneSide*2]
  nStates = nStatesOneSide*2+1
  
  # Agents
  alpha_MC = 2e-5
  gamma_MC = 1.0
  nParams_MC = 10
  approximationFunctionArgs_MC = {'af':linearTransform, 'afd':dLinearTransform, 
    'ftf':stateAggregation, 'nStates':nStates, 'nParams':nParams_MC}  
  
  env = RandomWalk(nStatesOneSide, specialRewards=specialRewards)
  agent_MC = GradientMonteCarloPrediction(env.nStates, nParams_MC, alpha_MC, gamma_MC, approximationFunctionArgs=approximationFunctionArgs_MC)
  
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
      
      state = new_state
      
    agent_MC.update(experiences)
  
  estimatedValues_MC = [agent_MC.getValue(state) for state in range(env.nStates-1)]

  pl.figure()
  pl.plot(groundTruth, 'k', label="Real values")
  pl.plot(estimatedValues_MC, label=agent_MC.getName())
  pl.xlabel("State")
  pl.ylabel("Value")
  pl.legend()
  
  mu = agent_MC.getMu()
  pl.figure()
  pl.bar(range(len(mu)), mu)
  pl.xlabel("State")
  pl.ylabel("Distribution")
  pl.show()

  