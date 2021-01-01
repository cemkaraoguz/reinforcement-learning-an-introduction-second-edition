'''
01_RandomWalk.py : replication of Figure 6.2

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.ToyExamples import RandomWalk
from IRL.agents.TemporalDifferenceLearning import TDPrediction
from IRL.agents.MonteCarlo import MonteCarloPrediction

if __name__=="__main__":

  nExperiments = 100
  nEpochs = 100

  # Environment
  nStatesOneSide = 3
  specialRewards = {nStatesOneSide*2:1.0}
  
  # Agent
  alphas_MC = [0.005]
  alphas_TD = [0.05, 0.005]
  doBatchUpdates_TD = [False, True]
  gamma = 1.0

  avg_rms_TD = np.zeros([nEpochs, len(alphas_TD)])
  avg_rms_MC = np.zeros([nEpochs, len(alphas_MC)])
  for idx_experiment in range(nExperiments):
    env = RandomWalk(nStatesOneSide, specialRewards=specialRewards)
    groundTruth = np.arange(1,env.nStates-1)/(env.nStates-1)
    # TD agents
    agents_TD = []
    valueTables_TD = []
    aux = []
    for alpha in alphas_TD:
      agent = TDPrediction(env.nStates, alpha, gamma)
      agent.valueTable = agent.valueTable + 0.5
      agent.valueTable[0] = 0.0
      agent.valueTable[nStatesOneSide*2] = 0.0
      agents_TD.append(agent)
      aux.append(np.array(agent.valueTable))    
    valueTables_TD.append(np.array(aux))
    # MC agents
    agents_MC = []
    valueTables_MC = []
    aux = []
    for alpha in alphas_MC:
      agent = MonteCarloPrediction(env.nStates, gamma, alpha)
      agent.valueTable = agent.valueTable + 0.5
      agent.valueTable[0] = 0.0
      agent.valueTable[nStatesOneSide*2] = 0.0
      agents_MC.append(agent)
      aux.append(np.array(agent.valueTable))
    valueTables_MC.append(np.array(aux))
    
    env.printEnv()
    
    allExperiences = []
    for e in range(nEpochs):
      
      print("Experiment:", idx_experiment, "Epoch : ", e)
      
      done = False
      experiences = [{}]
      state = env.reset()
      while not done:
        
        #print("State:", state, "Action: ", env.lastAction, "Reward: ", reward, "New state:", new_state)
        
        experiences[-1]['state'] = state
        experiences[-1]['done'] = done

        new_state, reward, done = env.step()
        
        xp = {}
        xp['reward'] = reward
        xp['state'] = new_state
        xp['done'] = done
        experiences.append(xp)
        
        state = new_state
        
        for i, agent in enumerate(agents_TD):
          if doBatchUpdates_TD[i]==False:
            agent.evaluate(experiences[-2:])

      allExperiences.append(experiences)

      for i, agent in enumerate(agents_TD):
        if doBatchUpdates_TD[i]==True:
          for ex in allExperiences:
            agent.evaluate(ex)
          
      for agent in agents_MC:
        for ex in allExperiences:
            agent.evaluate(ex)
            
      aux = []
      for i in range(len(agents_TD)):
        rms_TD = np.sqrt(np.mean((agents_TD[i].valueTable[1:env.nStates-1]-groundTruth)**2))
        avg_rms_TD[e,i] = avg_rms_TD[e,i] + (1.0/(idx_experiment+1)) * (rms_TD - avg_rms_TD[e,i])
        aux.append(np.array(agents_TD[i].valueTable))
      valueTables_TD.append(np.array(aux))
      
      aux = []
      for i in range(len(agents_MC)):
        rms_MC = np.sqrt(np.mean((agents_MC[i].valueTable[1:env.nStates-1]-groundTruth)**2))
        avg_rms_MC[e,i] = avg_rms_MC[e,i] + (1.0/(idx_experiment+1)) * (rms_MC - avg_rms_MC[e,i])
        aux.append(np.array(agents_MC[i].valueTable))
      valueTables_MC.append(np.array(aux))
  
  pl.figure()
  plotstyles_TD = ['-c','--c','-.c']
  plotstyles_MC = ['-r','--r','-.r', ':r']
  for i in range(len(agents_TD)):
    pl.plot(avg_rms_TD[:,i], plotstyles_TD[i], label="TD alpha="+str(alphas_TD[i])+", Batch update="+str(doBatchUpdates_TD[i]))
  for i in range(len(agents_MC)):
    pl.plot(avg_rms_MC[:,i], plotstyles_MC[i], label="MC alpha="+str(alphas_MC[i]))   
  pl.xlabel("Episodes")
  pl.ylabel("RMS Error, averaged over states")
  pl.legend()
  pl.show()

  