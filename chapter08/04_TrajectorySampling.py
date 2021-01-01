'''
04_TrajectorySampling.py : Replication of figure in Example 8.8 and solution to Exercise 8.8

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.ToyExamples import TrajectorySamplingTask
from IRL.agents.DynamicProgramming import ExpectedUpdateAgent, PolicyEvaluation
from IRL.utils.Policies import ActionValuePolicy

def runExperiment_US(nUpdatesMax, env, agent, agent_PE, evaluationFrequency):
  '''
  Uniform sampling
  '''
  values = []
  nUpdate = 0
  while True:
    for idx_state in range(env.nStates):
      for idx_action in range(env.nActions):
        agent.update(idx_state, idx_action)
        if nUpdate%evaluationFrequency==0:
          print("US update", nUpdate)
          values.append(evaluatePolicy(agent_PE, agent.policy, 100)[env.startState])
        nUpdate += 1
        if nUpdate>=nUpdatesMax:
          return np.array(values)
          
def runExperiment_TS(nUpdatesMax, env, agent, agent_PE, evaluationFrequency):
  '''
  Trajectory sampling
  '''
  values = []
  nUpdate = 0
  while nUpdate<nUpdatesMax:
    state = env.reset()
    done = False
    while not done and nUpdate < nUpdatesMax:
      action = agent.selectAction(state)
      agent.update(state, action)
      state, reward, done = env.step(action)
      if nUpdate%evaluationFrequency==0:
        print("TS update", nUpdate)
        values.append(evaluatePolicy(agent_PE, agent.policy, 100)[env.startState])
      nUpdate += 1
  return np.array(values) 

def evaluatePolicy(agent_PE, policy, maxEvalEpisodes):
  agent_PE.reset()
  for e in range(maxEvalEpisodes):
    deltaMax, isConverged = agent_PE.evaluate(policy)
    if(isConverged):
      break
  return agent_PE.valueTable
 
if __name__=="__main__":

  doSmallStateSpace = False
  
  if(doSmallStateSpace):
    # Figure 8.8, top
    nExperiments = 20
    nUpdatesMax = 20000
    evaluationFrequency = 1000
    
    # Environment
    nStates = 1000
    branchingFactors = [1, 3, 10]
  else:
    # Figure 8.8, bottom and Exercise 8.8
    nExperiments = 5
    nUpdatesMax = 200000
    evaluationFrequency = 10000
    
    # Environment
    nStates = 10000
    branchingFactors = [1, 3]
  
  # Agents
  gamma = 1.0
  epsilon = 0.1
  thresh_convergence = 1e-4

  values_all_US = []
  values_all_TS = []  
  for b in branchingFactors:
    env = TrajectorySamplingTask(nStates, b)
    values_avg_US = np.zeros(nUpdatesMax//evaluationFrequency)
    values_avg_TS = np.zeros(nUpdatesMax//evaluationFrequency)
    for idx_exp in range(nExperiments):
      
      print ("b:", b, "Experiment: ", idx_exp)
      
      agent_PE = PolicyEvaluation(env.nStates, env.nActions, gamma, thresh_convergence, env.computeExpectedValue)
      agent_US = ExpectedUpdateAgent(env.nStates, env.nActions, gamma, env.computeExpectedUpdate, epsilon=epsilon)   
      values = runExperiment_US(nUpdatesMax, env, agent_US, agent_PE, evaluationFrequency)
      values_avg_US = values_avg_US + (1.0/(idx_exp+1))*(values - values_avg_US)
      agent_TS = ExpectedUpdateAgent(env.nStates, env.nActions, gamma, env.computeExpectedUpdate, epsilon=epsilon)   
      values = runExperiment_TS(nUpdatesMax, env, agent_TS, agent_PE, evaluationFrequency)
      values_avg_TS = values_avg_TS + (1.0/(idx_exp+1))*(values - values_avg_TS)
      
    values_all_US.append(values_avg_US)
    values_all_TS.append(values_avg_TS)

  pl_clr = ['g', 'r', 'b']
  pl.figure()
  for idx_b, b in enumerate(branchingFactors):
    pl.plot(range(0,nUpdatesMax,evaluationFrequency), values_all_US[idx_b], '-'+pl_clr[idx_b], label="Uniform, b="+str(b))
    pl.plot(range(0,nUpdatesMax,evaluationFrequency), values_all_TS[idx_b], '--'+pl_clr[idx_b], label="On-policy, b="+str(b))
  pl.legend()
  pl.xlabel("Average computation time")
  pl.ylabel("Value of start state under greedy policy")
  pl.show()
