'''
01_RandomWalk_TDn.py : Replication of Figure 7.2

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.ToyExamples import RandomWalk
from IRL.agents.TemporalDifferenceLearning import nStepTDPrediction

def generateTrajectories(nEpisodes, env):
  trajectories_all = []
  for e in range(nEpisodes):
    done = False
    state = env.reset()
    trajectories = [{}]
    while not done:
      
      trajectories[-1]['state']= state
      trajectories[-1]['done']= done
      
      new_state, reward, done = env.step()
      
      #print("State:", state, "Action: ", env.lastAction, "Reward: ", reward, "New state:", new_state, "Done:", done)
      
      experience = {}
      experience['state'] = new_state
      experience['reward'] = reward
      experience['done'] = done
      trajectories.append(experience)
      
      state = new_state
      
    trajectories_all.append(trajectories)
    
  return trajectories_all

def runExperiment(trajectories, agent, groundTruth):
  for e, trajectory in enumerate(trajectories):
    for t in range(len(trajectory)-1):
      agent.evaluate(trajectory[t:t+2])
  rms = np.sqrt(np.mean((agent.valueTable[1:agent.nStates-1]-groundTruth)**2))
  return rms
  
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
  
  # Agent
  alphas = np.arange(0.0, 1.0+0.1, 0.1)
  nValues = [2**i for i in range(10)]
  gamma = 1.0

  env = RandomWalk(nStatesOneSide, specialRewards=specialRewards)

  trajectories = []
  for i in range(nExperiments):
    trajectories.append( generateTrajectories(nEpisodes, env) )
  
  avg_rms_all = np.zeros([len(nValues), len(alphas)])
  for idx_nValue, n in enumerate(nValues):
    for idx_alpha, alpha in enumerate(alphas):
      
      print("n: ", n, "alpha: ", alpha)

      avg_rms = 0.0
      for idx_experiment in range(nExperiments):
        agent = nStepTDPrediction(env.nStates, alpha, gamma, n)       
        rms = runExperiment(trajectories[idx_experiment], agent, groundTruth)
        avg_rms = avg_rms + (1.0/(idx_experiment+1)) * (rms - avg_rms)
      avg_rms_all[idx_nValue, idx_alpha] = avg_rms
  
  pl.figure()
  for i,n in enumerate(nValues):
    pl.plot(avg_rms_all[i,:], label="n="+str(n))
  pl.legend()
  #pl.ylim([0.25,0.55])
  pl.xticks(range(len(alphas)), [str(a)[0:3] for a in alphas])
  pl.xlabel("Alpha")
  pl.ylabel("Average RMS error over "+str(env.nStates)+" states and first "+str(nEpisodes)+" episodes")
  pl.show()

  