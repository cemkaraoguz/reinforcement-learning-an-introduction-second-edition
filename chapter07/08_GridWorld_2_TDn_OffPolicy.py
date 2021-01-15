'''
08_GridWorld_2_TDn_OffPolicy.py : Solution to Exercise 7.10

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.DynamicProgramming import PolicyEvaluation
from IRL.utils.Policies import StochasticPolicy
from IRL.agents.TemporalDifferenceLearning import nStepOffPolicyTDPrediction, nStepPerDecisionTDPrediction

def runExperiment(nEpisodes, env, agent, valueTable_ref, policy_behaviour=None):
  rms = np.zeros(nEpisodes)
  for e in range(nEpisodes):
    
    if(e%100==0):
      print("Episode : ", e)
      
    state = env.reset()
    if(policy_behaviour is None):
      action = agent.selectAction(state)
    else:
      action = policy_behaviour.sampleAction(state)     
    done = False
    experiences = [{}]
    while not done:
    
      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done
      
      new_state, reward, done = env.step(action)
      
      #print("Episode:", e, "State:", state, "Action: ", env.actionMapping[action][1], "Reward: ", reward, "New state:", new_state, "done:", done)
      
      if(policy_behaviour is None):
        new_action = agent.selectAction(new_state)
      else:
        new_action = policy_behaviour.sampleAction(new_state)  
      
      xp = {}
      xp['state'] = new_state
      xp['reward'] = reward
      xp['done'] = done
      xp['action'] = new_action
      experiences.append(xp)
      
      if(policy_behaviour is None):
        agent.evaluate(experiences[-2:])
      else:
        agent.evaluate(experiences[-2:], policy_behaviour)
    
      state = new_state
      action = new_action
      rms[e] = np.sqrt(np.mean((valueTable_ref - agent.valueTable)**2))
      
  return rms
  
if __name__=="__main__":

  nExperiments = 100
  nEpisodes = 100

  # Environment
  sizeX = 4
  sizeY = 4
  defaultReward = -1.0
  terminalStates= [(0,0), (3,3)]
  
  # Agent
  gamma = 0.9
  thresh_convergence = 1e-30
  n = 5
  alpha_TDnOP = 0.001
  alpha_TDnPD = 0.001
 
  env = DeterministicGridWorld(sizeX, sizeY, defaultReward=defaultReward, terminalStates=terminalStates)
  # Behaviour policy is a simple stochastic policy with equiprobable actions
  behaviour_policy = StochasticPolicy(env.nStates, env.nActions)
  # Load target policy q table
  # We will use the optimal policy learned via VI as target policy
  # These are the values learned in chapter04/03_GridWorld_2_VI.py
  with open('gridworld_2_qtable.npy', 'rb') as f:
    targetPolicy_qTable = np.load(f)  
  target_policy = StochasticPolicy(env.nStates, env.nActions)
  for s in range(env.nStates):
    target_policy.update(s, targetPolicy_qTable[s,:])
  # A policy evaluation agent will provide the ground truth
  agent_PE = PolicyEvaluation(env.nStates, env.nActions, gamma, thresh_convergence, env.computeExpectedValue)
  
  env.printEnv()
  
  # Policy evaluation for reference
  for e in range(nEpisodes):
      
    deltaMax, isConverged = agent_PE.evaluate(target_policy)
    
    #print("Episode : ", e, " Delta: ", deltaMax)
    
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
  
  avg_rms_tdnop = np.zeros(nEpisodes)
  avg_rms_tdnpd = np.zeros(nEpisodes)
  for idx_experiment in range(1, nExperiments+1):
  
    print("Experiment : ", idx_experiment)

    agent_TDnOP = nStepOffPolicyTDPrediction(env.nStates, env.nActions, alpha_TDnOP, gamma, n)
    agent_TDnOP.policy = target_policy
    agent_TDnPD = nStepPerDecisionTDPrediction(env.nStates, env.nActions, alpha_TDnPD, gamma, n)
    agent_TDnPD.policy = target_policy
    rms_tdnop = runExperiment(nEpisodes, env, agent_TDnOP, agent_PE.valueTable, behaviour_policy)
    rms_tdnpd = runExperiment(nEpisodes, env, agent_TDnPD, agent_PE.valueTable, behaviour_policy)
    avg_rms_tdnop = avg_rms_tdnop + (1.0/idx_experiment)*(rms_tdnop - avg_rms_tdnop)
    avg_rms_tdnpd = avg_rms_tdnpd + (1.0/idx_experiment)*(rms_tdnpd - avg_rms_tdnpd)
  
  pl.figure()
  pl.plot(agent_PE.valueTable, '-r', label="Policy Evaluation")
  pl.plot(agent_TDnOP.valueTable, '-b', label=str(n) + " step Off-policy TD prediction")
  pl.plot(agent_TDnPD.valueTable, '-k', label=str(n) + " step Per-decision TD prediction")
  pl.xlabel("States")
  pl.ylabel("Values")
  pl.legend()
  pl.figure()
  pl.plot(avg_rms_tdnop, '-b', label=str(n) + " step Off-policy TD prediction")
  pl.plot(avg_rms_tdnpd, '-k', label=str(n) + " step Per-decision TD prediction")
  pl.xlabel("Episodes")
  pl.ylabel("RMS of value estimates")
  pl.legend()
  pl.show()
  