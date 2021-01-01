'''
00_RandomWalk_alpha.py : Replication of figures 12.3, 12.6 and 12.8

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.ToyExamples import RandomWalk
from IRL.agents.EligibilityTraces import OfflineLambdaReturn, SemiGradientTDLambda, TrueOnlineTDLambda, OnlineLambdaReturn
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

      agent.evaluate(experiences)
      
      state = new_state

  estimatedValues = [agent.getValue(state) for state in range(env.nStates)]
  return np.array(estimatedValues)
  
if __name__=="__main__":  
  nExperiments = 100
  nEpisodes = 10
  
  # Environment
  nStatesOneSide = 9
  specialRewards = {nStatesOneSide*2:1.0, 0:-1.0}
  groundTruth = np.zeros(nStatesOneSide*2+1)
  groundTruth[nStatesOneSide:] = np.arange(nStatesOneSide+1)/nStatesOneSide
  groundTruth[0:nStatesOneSide] = np.arange(nStatesOneSide,0,-1)/(-nStatesOneSide)
  groundTruth = groundTruth[1:nStatesOneSide*2]
  nStates = nStatesOneSide*2+1
  
  # Agents
  alphas = np.arange(0.01, 1.01, 0.09)
  lambdaVals = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
  gamma = 1.0
  nParams = nStates
  approximationFunctionArgs = {'af':linearTransform, 'afd':dLinearTransform, 
    'ftf':stateAggregation, 'nStates':nStates, 'nParams':nParams}
  
  env = RandomWalk(nStatesOneSide, specialRewards=specialRewards) 
  agent_OffLR = OfflineLambdaReturn(nParams, 0.0, gamma, 0.0, approximationFunctionArgs=approximationFunctionArgs)
  agent_SGTDL = SemiGradientTDLambda(nParams, 0.0, gamma, 0.0, approximationFunctionArgs=approximationFunctionArgs)
  agent_TOGL = TrueOnlineTDLambda(nParams, 0.0, gamma, 0.0, approximationFunctionArgs=approximationFunctionArgs)
  agent_OnLR = OnlineLambdaReturn(nParams, 0.0, gamma, 0.0, approximationFunctionArgs=approximationFunctionArgs) # High complexity
  agents = [agent_OffLR, agent_SGTDL, agent_TOGL, agent_OnLR]
  agents = [agent_OffLR, agent_SGTDL, agent_TOGL]
  for agent in agents:
    avg_vals_all = []
    avg_rmse_all = []
    for lambd in lambdaVals:
      for alpha in alphas:
        avg_rmse = 0.0
        avg_vals = np.zeros(env.nStates)
        for idx_experiment in range(nExperiments):
          
          print(agent.getName(), "lambda", lambd, "alpha:", alpha, "idxExperiment:", idx_experiment)
          
          # OfflineLambdaReturn
          agent.reset()
          agent.alpha = alpha
          agent.lambd = lambd
          estimatedValues = runExperiment(nEpisodes, env, agent)
          rmse = np.sqrt(np.mean( (groundTruth - estimatedValues[1:env.nStates-1])**2 ) )
          avg_vals = avg_vals + (1.0/(idx_experiment+1))*(estimatedValues - avg_vals)
          avg_rmse = avg_rmse + (1.0/(idx_experiment+1))*(rmse - avg_rmse)
          
          print("Average RMS for ", agent.getName(), " for lambda:", lambd, " is ", rmse)
          
        avg_vals_all.append(avg_vals)
        avg_rmse_all.append(avg_rmse)
        
    fig, ax = pl.subplots()
    for i, lambd in enumerate(lambdaVals):
      ax.plot(avg_rmse_all[i*len(alphas):i*len(alphas)+len(alphas)], label="lambda="+str(lambd))
    ax.set_xlabel("alpha")
    ax.set_ylabel("Average RMS Error")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([str(np.round(i,1)) for i in alphas])
    ax.set_title(agent.getName())
    ax.set_ylim([0.0, 0.55])
    pl.legend()
    for j, lambd in enumerate(lambdaVals):
      pl.figure()
      pl.title(agent.getName()+" lambda="+str(lambd))
      pl.plot(groundTruth, 'k', label="Real values")
      for i, alpha in enumerate(alphas):
        pl.plot(np.array(avg_vals_all[j*len(alphas)+i]), label="alpha="+str(round(alpha,1)))
      pl.xlabel("State")
      pl.ylabel("Value")
      pl.legend()
  pl.show()
