'''
01_MountainCar_SARSA_alphas.py : Replication of figures 12.10 and 12.11

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from IRL.environments.Cars import MountainCar
from IRL.agents.EligibilityTraces import SARSALambda, TrueOnlineSARSA
from IRL.utils.ApproximationFunctions import linearTransform, dLinearTransform
from IRL.utils.FeatureTransformations import tileCoding

def runExperiment(nEpisodes, env, agent, nStepsMax):
  nStepsPerEpisode = np.zeros(nEpisodes)
  rewards = np.zeros(nEpisodes)
  for e in range(nEpisodes):
    
    if e%1==0:
      print("Episode : ", e)
    
    experiences = [{}]
    done = False
    state = env.reset()
    action = agent.selectAction(state)
    t = 0
    cum_reward = 0.0

    while not done:     

      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done

      new_state, reward, done = env.step(action)
      
      #print("Episode:", e, "State:", state, "Action: ", env.actionMapping[action][1], "Reward: ", reward, "New state:", new_state, "done:", done)

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
      cum_reward+=reward
      t += 1
      if t>=nStepsMax:
        break
      
    nStepsPerEpisode[e] = t
    rewards[e] = cum_reward
  
  return nStepsPerEpisode, rewards
  
if __name__=="__main__":

  nExperiments = 100
  nEpisodes = 20
  nStepsMax = 1000
  
  # Environment
  positionBounds = [-1.2, 0.5]
  velocityBounds = [-0.07, 0.07]
  startPositionBounds = [-0.6, -0.4]
  
  # Agent
  alphas = [0.2/8, 0.5/8, 0.8/8, 1.1/8, 1.4/8, 1.8/8]
  lambdaVals = [0.0, 0.68, 0.84, 0.92, 0.96, 0.98, 0.99]
  gamma = 1.0
  epsilon = 0.0
  nActions = 3
  minStates = [positionBounds[0], velocityBounds[0]]
  maxStates = [positionBounds[1], velocityBounds[1]]
  nTilings = 8
  tilingOffsets = [[i, j] for i, j in zip(np.linspace(-0.4, 0.4,num=nTilings), np.linspace(-0.04, 0.04,num=nTilings))] # (idxTiling, dimState)
  tilingSize = [[8, 8] for _ in range(nTilings)]  # (idxTiling, dimState)
  nParams = nActions * np.sum([np.prod(i) for i in tilingSize])
  approximationFunctionArgs = {'af':linearTransform, 'afd':dLinearTransform, 'ftf':tileCoding,
    'minStates':minStates, 'maxStates':maxStates, 'nTilings':nTilings, 
    'tilingOffsets':tilingOffsets, 'tilingSize':tilingSize, 'nActions':nActions}

  env = MountainCar(positionBounds, velocityBounds, startPositionBounds)
  agent_SARSALambda_rep = SARSALambda(nParams, nActions, 0.0, gamma, 0.0, 
    approximationFunctionArgs=approximationFunctionArgs, doAccumulateTraces=False, doClearTraces=False, epsilon=epsilon)
  agent_SARSALambda_acc = SARSALambda(nParams, nActions, 0.0, gamma, 0.0, 
    approximationFunctionArgs=approximationFunctionArgs, doAccumulateTraces=True, doClearTraces=False, epsilon=epsilon)
  agent_SARSALambda_clr = SARSALambda(nParams, nActions, 0.0, gamma, 0.0, 
    approximationFunctionArgs=approximationFunctionArgs, doAccumulateTraces=False, doClearTraces=True, epsilon=epsilon)
  agent_TrueOnlineSARSA = TrueOnlineSARSA(nParams, nActions, 0.0, gamma, 0.0, 
    approximationFunctionArgs=approximationFunctionArgs, epsilon=epsilon)
  
  agents = [agent_SARSALambda_rep, agent_SARSALambda_acc, agent_SARSALambda_clr, agent_TrueOnlineSARSA]
  labels = ["SARSA(lambda) Replacing traces", "SARSA(lambda) Accumulating traces", "SARSA(lambda) Clearing traces", "True Online SARSA"]
  rewards_all = []
  nStepsPerEpisode_avg_all = []
  for agent in agents:
    for lambd in lambdaVals:
      for alpha in alphas:
        nStepsPerEpisode_avg = 0.0
        rewards_avg = 0.0
        for idx_experiment in range(1, nExperiments+1):
          
          print(agent.getName(), "lambda:", lambd, "alpha:", alpha, "idxExperiment", idx_experiment)
          
          agent.reset()
          agent.alpha = alpha
          agent.lambd = lambd         
          nStepsPerEpisode, rewards = runExperiment(nEpisodes, env, agent, nStepsMax)
          nStepsPerEpisode_avg = nStepsPerEpisode_avg + (1.0/idx_experiment)*(np.mean(nStepsPerEpisode) - nStepsPerEpisode_avg)
          rewards_avg = rewards_avg + (1.0/idx_experiment)*(np.mean(rewards) - rewards_avg)
        nStepsPerEpisode_avg_all.append(nStepsPerEpisode_avg)
        rewards_all.append(rewards_avg)
    
  for idx_agent, agent in enumerate(agents):
    fig, ax = pl.subplots()
    aux0 = nStepsPerEpisode_avg_all[idx_agent*len(lambdaVals)*len(alphas):(idx_agent+1)*len(lambdaVals)*len(alphas)]
    for i, lambd in enumerate(lambdaVals):
      plotarray = np.flip(aux0[i*len(alphas):i*len(alphas)+len(alphas)])
      ax.plot(plotarray, label="lambda="+str(lambd))
    ax.set_xlabel("alpha")
    ax.set_ylabel("Number of steps per episode averaged over "+str(nExperiments)+" experiments")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([str(np.round(i,4)) for i in alphas])
    ax.set_title(labels[idx_agent])
    pl.legend()
  for idx_lambda_vis, lambd in enumerate(lambdaVals):
    fig, ax = pl.subplots()
    for idx_agent, agent in enumerate(agents):
      aux0 = rewards_all[idx_agent*len(lambdaVals)*len(alphas):(idx_agent+1)*len(lambdaVals)*len(alphas)]
      aux1 = aux0[idx_lambda_vis*len(alphas):(idx_lambda_vis+1)*len(alphas)]
      ax.plot(aux1, label=labels[idx_agent])
    ax.set_xlabel("alpha")
    ax.set_ylabel("Reward per episode averaged over first "+str(nEpisodes)+" episodes and "+str(nExperiments)+" experiments")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([str(np.round(k,4)) for k in alphas])
    ax.set_title("lambda="+str(lambd))
    pl.legend() 
  pl.show() 