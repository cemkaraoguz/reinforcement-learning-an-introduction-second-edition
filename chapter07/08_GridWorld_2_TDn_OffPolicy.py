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
from IRL.agents.TemporalDifferenceLearning import nStepTDPrediction, nStepOffPolicyTDPrediction, nStepPerDecisionTDPrediction

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
        if(agent.name=="n-step TD Prediction"):
          agent.evaluate(experiences[-2:])
        else:
          agent.evaluate(experiences[-2:], policy_behaviour)
    
      state = new_state
      action = new_action
      rms[e] = np.sqrt(np.mean((valueTable_ref - agent.valueTable)**2))
      
  return rms
  
if __name__=="__main__":

  nExperiments = 100
  nEpisodes = 400

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
  alpha_TDnOP = 0.01
  alpha_TDnPD = 0.01
  
  env = DeterministicGridWorld(sizeX, sizeY, defaultReward=defaultReward, terminalStates=terminalStates)
  policy = StochasticPolicy(env.nStates, env.nActions)
  agent_PE = PolicyEvaluation(env.nStates, env.nActions, gamma, thresh_convergence, env.computeExpectedValue)
  
  env.printEnv()
  
  # Policy evaluation for reference
  for e in range(nEpisodes):
      
    deltaMax, isConverged = agent_PE.evaluate(policy)
    
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
  
  avg_rms_tdn = np.zeros(nEpisodes)
  avg_rms_tdnop = np.zeros(nEpisodes)
  avg_rms_tdnpd = np.zeros(nEpisodes)
  for idx_experiment in range(1, nExperiments+1):
  
    print("Experiment : ", idx_experiment)

    agent_TDn = nStepTDPrediction(env.nStates, alpha_TDn, gamma, n)
    agent_TDnOP = nStepOffPolicyTDPrediction(env.nStates, env.nActions, alpha_TDnOP, gamma, n)
    agent_TDnPD = nStepPerDecisionTDPrediction(env.nStates, env.nActions, alpha_TDnPD, gamma, n)

    rms_tdn = runExperiment(nEpisodes, env, agent_TDn, agent_PE.valueTable, policy)
    rms_tdnop = runExperiment(nEpisodes, env, agent_TDnOP, agent_PE.valueTable, policy)
    rms_tdnpd = runExperiment(nEpisodes, env, agent_TDnPD, agent_PE.valueTable, policy)

    avg_rms_tdn = avg_rms_tdn + (1.0/idx_experiment)*(rms_tdn - avg_rms_tdn)
    avg_rms_tdnop = avg_rms_tdnop + (1.0/idx_experiment)*(rms_tdnop - avg_rms_tdnop)
    avg_rms_tdnpd = avg_rms_tdnpd + (1.0/idx_experiment)*(rms_tdnpd - avg_rms_tdnpd)
  
  pl.figure()
  pl.plot(agent_PE.valueTable, '-r', label="Policy Evaluation")
  pl.plot(agent_TDn.valueTable, '-g', label=str(n) + " step TD prediction")
  pl.plot(agent_TDnOP.valueTable, '-b', label=str(n) + " step Off-policy TD prediction")
  pl.plot(agent_TDnPD.valueTable, '-k', label=str(n) + " step Per-decision TD prediction")
  pl.xlabel("States")
  pl.ylabel("Values")
  pl.legend()
  pl.figure()
  pl.plot(avg_rms_tdn, '-g', label=str(n) + " step TD prediction")
  pl.plot(avg_rms_tdnop, '-b', label=str(n) + " step Off-policy TD prediction")
  pl.plot(avg_rms_tdnpd, '-k', label=str(n) + " step Per-decision TD prediction")
  pl.xlabel("Episodes")
  pl.ylabel("RMS of value estimates")
  pl.legend()
  pl.show()
  