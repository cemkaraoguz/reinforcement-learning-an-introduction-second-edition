'''
00_GridWorld_2_TDn.py : Solution to exercise 7.2

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.DynamicProgramming import PolicyEvaluation
from IRL.utils.Policies import StochasticPolicy
from IRL.agents.TemporalDifferenceLearning import TDPrediction, nStepTDPrediction
  
def update_SumTDError(valueTable, experiences, n, alpha, gamma):
  for timestep in range(len(experiences)+n):
    t = timestep - n
    if(t<0):
      continue
    sum_td_error = 0
    for i in range(t, min(t+n,len(experiences))-1):
      state = experiences[i]['state']
      reward = experiences[i+1]['reward']
      next_state = experiences[i+1]['state']
      td_error = reward + gamma * valueTable[next_state] - valueTable[state]
      sum_td_error = sum_td_error + gamma ** (i-t) * td_error
    valueTable[experiences[t]['state']] = valueTable[experiences[t]['state']] + alpha * sum_td_error
  return valueTable
  
def normalize_(values):
  values_pos = values-min(values)
  return values_pos/max(values_pos)

def normalize(values):
  return values
  
if __name__=="__main__":

  nEpisodes = 1000

  # Environment
  sizeX = 4
  sizeY = 4
  defaultReward = -1.0
  terminalStates= [(0,0), (3,3)]

  # Agent
  gamma = 1.0
  thresh_convergence = 1e-30
  n = 5
  alpha_TDn = 0.01
  alpha_TD = 0.01
  alpha_sumTDError = 0.01
  
  env = DeterministicGridWorld(sizeX, sizeY, defaultReward=defaultReward, terminalStates=terminalStates)
  policy = StochasticPolicy(env.nStates, env.nActions)
  agent_PE = PolicyEvaluation(env.nStates, env.nActions, gamma, thresh_convergence, env.computeExpectedValue)
  
  # TD agent to validate the TDn implementation
  agent_TD = TDPrediction(env.nStates, alpha_TD, gamma)
  agent_TDn = nStepTDPrediction(env.nStates, alpha_TDn, gamma, n)

  env.printEnv()
  
  # Policy evaluation for reference
  for e in range(nEpisodes):
    deltaMax, isConverged = agent_PE.evaluate(policy)
    
    print("Episode : ", e, " Delta: ", deltaMax)
    
    printStr = ""
    for y in range(sizeY):
      for x in range(sizeX):
        i = env.getLinearIndex(x,y)
        printStr += "{:.2f}".format(agent_PE.valueTable[i]) + "\t"
      printStr += "\n"
    print(printStr)
    
    if(isConverged):
      print("Convergence achieved!")
      break
  
  # TD(n) prediction vs averaging TD errors
  values_sumTDerrors = np.zeros(env.nStates)
  
  for e in range(nEpisodes):
  
    if(e%int(nEpisodes*0.1)==0):
      print("Episode : ", e)
    
    experiences = [{}]
    state = env.reset()
    action = policy.sampleAction(state)
    done = False
    while not done:
      
      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done
      
      new_state, reward, done = env.step(action)

      #print("Episode : ", e, " State : ", state, " Action : ", action, " Reward : ", reward, " Next state : ", new_state)
      
      new_action = policy.sampleAction(state)
      
      xp = {}
      xp['state'] = new_state
      xp['reward'] = reward
      xp['done'] = done
      xp['action'] = new_action
      experiences.append(xp)
      
      agent_TDn.evaluate(experiences[-2:])
      agent_TD.evaluate(experiences[-2:])

      state = new_state
      action = new_action
    
    values_sumTDerrors = update_SumTDError(values_sumTDerrors, experiences, n, alpha_sumTDError, gamma)
    
  pl.figure()
  pl.plot(normalize(agent_PE.valueTable), label="Policy Evaluation")
  pl.plot(normalize(agent_TD.valueTable), label="1 step TD Prediction") 
  pl.plot(normalize(agent_TDn.valueTable), label=str(n) + " step TD prediction")
  pl.plot(normalize(values_sumTDerrors), label="Sum of TD Errors")  
  pl.xlabel("States")
  pl.ylabel("Values (normalized)")
  pl.legend()
  pl.show()