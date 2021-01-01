'''
05_InfiniteVariance.py : Replication of Figure 5.4

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from IRL.environments.ToyExamples import InfiniteVariance
from IRL.agents.MonteCarlo import MonteCarloOffPolicyPrediction
from IRL.utils.Policies import StochasticPolicy

def runExperiment(nEpisodes, env, behaviour_policy, agent):
  valueEstimates = []
  for e in range(nEpisodes):
    
    if(e%10000==0):
     print("Episode : ", e)

    state = env.reset()
    experiences = [{}]
    done = False
    while not done:
    
      action = behaviour_policy.sampleAction(state)
      
      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done
      
      new_state, reward, done = env.step(action)

      #print("Epoch : ", e, " State : ", state, " Action : ", action, " Reward : ", reward, " Next state : ", new_state)
      
      xp = {}
      xp['reward'] = reward
      xp['state'] = new_state
      xp['done'] = done
      experiences.append(xp)

      state = new_state

    agent.evaluate(experiences, behaviour_policy)
    valueEstimates.append(agent.getValue(env.STATE_S))
    
  return np.array(valueEstimates)
  
if __name__=="__main__":

  nExperiments = 10
  nEpisodes = 1000000
  
  # Agent
  gamma = 1.0
  
  # Environment
  env = InfiniteVariance()
  behaviour_policy = StochasticPolicy(env.nStates, env.nActions)
  
  #env.printEnv()
  
  valueEstimations = np.zeros([nExperiments, nEpisodes])
  for idx_experiment in range(nExperiments):
    
    print("Experiment : ", idx_experiment)

    agent_ordinary = MonteCarloOffPolicyPrediction(env.nStates, env.nActions, gamma, doUseWeightedIS=False)
    actionProb = np.zeros(env.nActions)
    actionProb[env.ACTION_LEFT] = 1.0
    agent_ordinary.policy.update(env.STATE_S, actionProb)
    actionProb = np.zeros(env.nActions)
    agent_ordinary.policy.update(env.STATE_TERMINAL, actionProb)
    valueEstimations_experiment = runExperiment(nEpisodes, env, behaviour_policy, agent_ordinary)
    valueEstimations[idx_experiment,:] = valueEstimations_experiment[:]
  
  fig = pl.figure()
  for idx_experiment in range(nExperiments):
    pl.plot(valueEstimations[idx_experiment,:])
  pl.xlabel("Episodes")
  pl.ylabel("Monte Carlo Estimate of s with ordinary importance sampling")
  pl.xscale("log")
  pl.show()
  