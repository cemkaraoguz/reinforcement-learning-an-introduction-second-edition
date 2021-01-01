'''
06_RaceTrack_MCC_OffP.py : Solution to exercise 5.12

Cem Karaoguz, 2020
MIT License
'''

import sys
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from IRL.environments.Gridworlds import RaceTrack
from IRL.agents.MonteCarlo import MonteCarloOffPolicyControl
from IRL.utils.Policies import StochasticPolicy
from IRL.utils.Helpers import runSimulation

if __name__=="__main__":
  
  # General
  nEpochs = 10000

  # Environment
  trackID = 1
  defaultReward = -1.0
  outOfTrackReward = -1.0
  finishReward = 10.0
  p_actionFail = 0.1

  # Agent
  gamma = 1.0

  # Policy
  doUpdateBehaviourPolicy = True
  epsilon = 0.01

  if(trackID==0):
    sizeX = 17
    sizeY = 32
    startStates = [(x,31) for x in range(3,9)]
    terminalStates = [(16,y) for y in range(0,6)]
    outOfTrackStates = [(0,0), (1,0), (2,0), (0,1), (1,1), (0,2), (1,2), (0,3)]
    outOfTrackStates.extend([(0 ,y) for y in range(14,32)])
    outOfTrackStates.extend([(1 ,y) for y in range(22,32)])
    outOfTrackStates.extend([(2 ,y) for y in range(29,32)])
    outOfTrackStates.extend([(x ,6) for x in range(10,17)])
    outOfTrackStates.extend([(x ,6) for x in range(10,17)])
    outOfTrackStates.extend([(x ,y) for x in range(9,17) for y in range(7,32)])
  elif(trackID==1):
    sizeX = 32
    sizeY = 30
    startStates = [(x,29) for x in range(0,23)]
    terminalStates = [(31,y) for y in range(0,9)]
    outOfTrackStates = [(x  ,0) for x in range(0,16)]
    outOfTrackStates.extend([(x ,1) for x in range(0,13)])
    outOfTrackStates.extend([(x ,2) for x in range(0,12)])
    outOfTrackStates.extend([(x ,y) for x in range(0,11) for y in range(3,7)])
    outOfTrackStates.extend([(x ,7) for x in range(0,12)])
    outOfTrackStates.extend([(x ,8) for x in range(0,13)])
    outOfTrackStates.extend([(x ,y) for x in range(0,14) for y in range(9,14)])
    outOfTrackStates.extend([(x ,14) for x in range(0,13)])
    outOfTrackStates.extend([(x ,15) for x in range(0,12)])
    outOfTrackStates.extend([(x ,16) for x in range(0,11)])
    outOfTrackStates.extend([(x ,17) for x in range(0,10)])
    outOfTrackStates.extend([(x ,18) for x in range(0,9)])
    outOfTrackStates.extend([(x ,19) for x in range(0,8)])
    outOfTrackStates.extend([(x ,20) for x in range(0,7)])
    outOfTrackStates.extend([(x ,21) for x in range(0,6)])
    outOfTrackStates.extend([(x ,22) for x in range(0,5)])
    outOfTrackStates.extend([(x ,23) for x in range(0,4)])
    outOfTrackStates.extend([(x ,24) for x in range(0,3)])
    outOfTrackStates.extend([(x ,25) for x in range(0,2)])
    outOfTrackStates.extend([(x ,26) for x in range(0,1)])
    outOfTrackStates.extend([(x ,9) for x in range(30,32)])
    outOfTrackStates.extend([(x ,10) for x in range(27,32)])
    outOfTrackStates.extend([(x ,11) for x in range(26,32)])
    outOfTrackStates.extend([(x ,12) for x in range(24,32)])
    outOfTrackStates.extend([(x ,y) for x in range(23,32) for y in range(13,30)])
  else:
    sys.exit("ERROR: trackID not recognized")
    
  env = RaceTrack(sizeX, sizeY, startStates=startStates, terminalStates=terminalStates, 
    impassableStates=outOfTrackStates, defaultReward=defaultReward, crashReward=outOfTrackReward,
    finishReward=finishReward, p_actionFail=p_actionFail)
  agent = MonteCarloOffPolicyControl(env.nStates, env.nActions, gamma)
  behaviour_policy = StochasticPolicy(env.nStates, env.nActions, policyUpdateMethod="esoft", epsilon=epsilon)
  
  for e in range(nEpochs):
    
    if(e%1000==0):
      print("Epoch : ", e)

    experiences = [{}]
    state = env.reset()
    done = False
    while not done:
    
      action = behaviour_policy.sampleAction(state, env.getAvailableActions())

      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done
      experiences[-1]['allowedActions'] = env.getAvailableActions(state)# TODO check
      
      new_state, reward, done = env.step(action)

      xp = {}
      xp['reward'] = reward
      xp['state'] = new_state
      xp['allowedActions'] = env.getAvailableActions(new_state)# TODO check
      xp['done'] = done
      experiences.append(xp)

      state = new_state
    
    agent.update(experiences, behaviour_policy)

    if(doUpdateBehaviourPolicy):
      # update behaviour policy to be e-soft version of the target policy
      for idx_state in range(env.nStates):
        behaviour_policy.update(idx_state, agent.actionValueTable[idx_state,:])

  # Simulation after learning
  # -------------------------
  env.printEnv(agent)

  input("Press any key to continue...")

  env.p_actionFail = 0.0
  agentHistory = runSimulation(env, agent)
  
  print("Simulation:")
  
  env.render(agentHistory)
