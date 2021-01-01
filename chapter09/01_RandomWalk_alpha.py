'''
01_RandomWalk_alpha.py : Replication of Figure 9.2

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.ToyExamples import RandomWalk
from IRL.agents.TemporalDifferenceApproximation import nStepSemiGradientTDPrediction
from IRL.utils.FeatureTransformations import stateAggregation
from IRL.utils.ApproximationFunctions import linearTransform, dLinearTransform

def runExperiment(nEpisodes, env, agent):
  for e in range(nEpisodes):
    experiences = [{}]
    done = False
    state = env.reset()
    while not done:     
      experiences[-1]['state'] = state
      experiences[-1]['done'] = done

      new_state, reward, done = env.step()
      
      xp = {}
      xp['reward'] = reward
      xp['state'] = new_state
      xp['done'] = done
      experiences.append(xp)

      agent.update(experiences)
      
      state = new_state

  estimatedValues = [agent.getValue(state) for state in range(env.nStates)]
  return np.array(estimatedValues)
  
if __name__=="__main__":  
  nExperiments = 10
  nEpisodes = 10
  
  # Environment
  nStatesOneSide = 500
  specialRewards = {nStatesOneSide*2:1.0, 0:-1.0}
  groundTruth = np.zeros(nStatesOneSide*2+1)
  groundTruth[nStatesOneSide:] = np.arange(nStatesOneSide+1)/nStatesOneSide
  groundTruth[0:nStatesOneSide] = np.arange(nStatesOneSide,0,-1)/(-nStatesOneSide)
  groundTruth = groundTruth[1:nStatesOneSide*2]
  nStates = nStatesOneSide*2+1
  
  # Agents
  alphas = np.arange(0.01, 1.01, 0.19)
  nVals = [2**n for n in range(10)]
  gamma = 1.0
  nParams = 10
  approximationFunctionArgs_TD = {'af':linearTransform, 'afd':dLinearTransform, 
    'ftf':stateAggregation, 'nStates':nStates, 'nParams':nParams}
  
  env = RandomWalk(nStatesOneSide, specialRewards=specialRewards) 
  
  avg_vals_all = []
  avg_rmse_TDn_all = []
  for n in nVals:
    for alpha in alphas:
      avg_rmse_TDn = 0.0
      avg_vals_TDn = np.zeros(env.nStates)
      for idx_experiment in range(nExperiments):
        
        print("n", n, "alpha:", alpha, "idxExperiment:", idx_experiment)
        
        agent_TDn = nStepSemiGradientTDPrediction(nParams, alpha, gamma, n, approximationFunctionArgs=approximationFunctionArgs_TD)
        
        estimatedValues_TDn = runExperiment(nEpisodes, env, agent_TDn)
        rmse_TDn = np.sqrt(np.mean( (groundTruth - estimatedValues_TDn[1:env.nStates-1])**2 ) )
        avg_vals_TDn = avg_vals_TDn + (1.0/(idx_experiment+1))*(estimatedValues_TDn - avg_vals_TDn)
        avg_rmse_TDn = avg_rmse_TDn + (1.0/(idx_experiment+1))*(rmse_TDn - avg_rmse_TDn)
        
        print("Average RMS for ", n, " step TD approxiamtion is ", rmse_TDn)
      
      avg_vals_all.append(avg_vals_TDn)
      avg_rmse_TDn_all.append(avg_rmse_TDn)
    
  fig, ax = pl.subplots()
  for i, n in enumerate(nVals):
    ax.plot(avg_rmse_TDn_all[i*len(alphas):i*len(alphas)+len(alphas)], label=str(n)+" step TD")
  ax.set_xlabel("alpha")
  ax.set_ylabel("Average RMS Error")
  ax.set_xticks(range(len(alphas)))
  ax.set_xticklabels([str(np.round(i,1)) for i in alphas])
  pl.legend()
  
  for j, n in enumerate(nVals):
    pl.figure()
    pl.title("n="+str(n))
    pl.plot(groundTruth, 'k', label="Real values")
    for i, alpha in enumerate(alphas):
      pl.plot(np.array(avg_vals_all[j*len(alphas)+i]), label="alpha="+str(round(alpha,1)))
    pl.xlabel("State")
    pl.ylabel("Value")
    pl.legend()
  pl.show()
