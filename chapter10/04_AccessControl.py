'''
04_AccessControl.py : Replication of Figure 10.5

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from IRL.environments.ResourceAllocationTasks import AccessControlTask
from IRL.agents.TemporalDifferenceApproximation import DifferentialSemiGradientSARSA, DifferentialSemiGradientQLearning, nStepDifferentialSemiGradientSARSA
from IRL.utils.ApproximationFunctions import linearTransform, dLinearTransform
from IRL.utils.FeatureTransformations import tileCoding

def runExperiment(maxTimesteps, env, agent):

  experiences = [{}]
  done = False
  state = env.reset()
  action = agent.selectAction(state)

  for t in range(maxTimesteps):
    
    if t%10000==0:
      print("Timestep : ", t)
    
    experiences[-1]['state'] = state
    experiences[-1]['action'] = action
    experiences[-1]['done'] = done

    new_state, reward, done = env.step(action)
    
    #print("Timestep:", t, "State:", state, "Action: ", env.actionMapping[action][1], "Reward: ", reward, "New state:", new_state, "done:", done)
    
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
  
if __name__=="__main__":

  maxTimesteps = 1000000
  
  # Environment
  nServers = 10
  customerPriorities = [1,2,4,8]
  pServerFree = 0.06
  rejectionReward = 0.0
  
  env = AccessControlTask(nServers=nServers, customerPriorities=customerPriorities, pServerFree=pServerFree, rejectionReward=rejectionReward)

  # Agent
  alpha = 0.01
  beta = 0.01
  epsilon = 0.1
  n = 8
  
  minStates = [0, 0]
  maxStates = [env.dimStates[0], env.dimStates[1]]
  nTilings = 1
  tilingOffsets = [[0, 0]] # (idxTiling, dimState)
  tilingSize = [[env.dimStates[0], env.dimStates[1]]]  # (idxTiling, dimState)
  nParams = env.nActions * np.sum([np.prod(i) for i in tilingSize])
  approximationFunctionArgs = {'af':linearTransform, 'afd':dLinearTransform, 'ftf':tileCoding,
    'minStates':minStates, 'maxStates':maxStates, 'nTilings':nTilings, 
    'tilingOffsets':tilingOffsets, 'tilingSize':tilingSize, 'nActions':env.nActions}

  agent_DSGSARSA = DifferentialSemiGradientSARSA(nParams, env.nActions, alpha, beta, 
    approximationFunctionArgs=approximationFunctionArgs, epsilon=epsilon)
  agent_DSGQ = DifferentialSemiGradientQLearning(nParams, env.nActions, alpha, beta, 
    approximationFunctionArgs=approximationFunctionArgs, epsilon=epsilon)
  agent_nStepDSGSARSA = nStepDifferentialSemiGradientSARSA(nParams, env.nActions, alpha, beta, n, 
    approximationFunctionArgs=approximationFunctionArgs, epsilon=epsilon)

  agents = [agent_DSGSARSA, agent_DSGQ, agent_nStepDSGSARSA]
  for agent in agents:
    runExperiment(maxTimesteps, env, agent)
    maxValTable = np.zeros([len(customerPriorities), nServers])
    policyTable = np.zeros([len(customerPriorities), nServers])
    for idx_priority in range(len(customerPriorities)):
      for idx_nFreeServers in range(nServers):
        q = [agent.getValue([idx_priority, idx_nFreeServers], action) for action in range(env.nActions)]
        maxValTable[idx_priority, idx_nFreeServers] = max(q)
        policyTable[idx_priority, idx_nFreeServers] = np.argmax(q)
    
    pl.figure()
    pl.plot(np.zeros(nServers), '--k')
    for idx_priority in range(len(customerPriorities)):
      pl.plot(maxValTable[idx_priority,:], label="priority "+str(customerPriorities[idx_priority]))
    pl.xlabel('Number of free servers')
    pl.ylabel('Differential value of best action')
    pl.title(agent.getName())
    pl.legend()
    pl.show()
    
    print("Policy learned by agent ", agent.getName())
    for idx_priority in range(len(customerPriorities)):
      printStr = ""
      for idx_nFreeServers in range(nServers):
        printStr += env.actionMapping[policyTable[idx_priority, idx_nFreeServers]][1]+"\t"
      print(printStr)
      
  