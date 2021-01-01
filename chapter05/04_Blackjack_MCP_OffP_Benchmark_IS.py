'''
04_Blackjack_MCP_OffP_Benchmark_IS.py : Replication of Figure 5.3

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from IRL.environments.Gambler import Blackjack
from IRL.agents.MonteCarlo import MonteCarloOffPolicyPrediction
from IRL.utils.Policies import StochasticPolicy

def runExperiment(nEpisodes, env, behaviour_policy, agent):
  errors = []
  for e in range(nEpisodes):
    
    if(e%1000==0):
      print("Episode : ", e)

    _ = env.reset()
    #startState_dealerHand = [env.LABEL_ACE, min(env.VAL_FACECARDS, np.random.choice(env.deck))]
    startState_dealerHand = [env.LABEL_ACE, env.LABEL_ACE]
    startState_playerHand = [env.LABEL_ACE, env.LABEL_ACE, env.LABEL_ACE]
    startState = env.setHands(startState_playerHand, startState_dealerHand)
    state = startState
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
    
    errors.append( agent.getValue(startState) - startState_groundTruth )
    
  return np.array(errors)
  
if __name__=="__main__":

  nExperiments = 100
  nEpisodes = 10000
  
  # Agent
  gamma = 1.0
  
  # Environment
  env = Blackjack()

  #startState_dealerHand = [env.LABEL_ACE, min(env.VAL_FACECARDS, np.random.choice(env.deck))]
  startState_dealerHand = [env.LABEL_ACE, env.LABEL_ACE]
  startState_playerHand = [env.LABEL_ACE, env.LABEL_ACE, env.LABEL_ACE]
  startState = env.setHands(startState_playerHand, startState_dealerHand)
  startState_groundTruth = -0.27726

  behaviour_policy = StochasticPolicy(env.nStates, env.nActions)
  
  mse_weighted = np.zeros(nEpisodes)
  mse_ordinary = np.zeros(nEpisodes)
  for idx_experiment in range(nExperiments):
    agent_ordinary = MonteCarloOffPolicyPrediction(env.nStates, env.nActions, gamma, doUseWeightedIS=False)
    agent_weighted = MonteCarloOffPolicyPrediction(env.nStates, env.nActions, gamma, doUseWeightedIS=True)
    for i in range(env.nStatesPlayerSum-1, -1, -1):
      for j in range(env.nStatesDealerShowing):
        for k in [env.USABLE_ACE_YES, env.USABLE_ACE_NO]:
          idx_state = env.getLinearIndex(env.minPlayerSum+i, env.minDealerShowing+j, k)
          if(env.minPlayerSum+i<20):
            actionProb = np.zeros(env.nActions)
            actionProb[env.ACTION_HIT] = 1.0
            agent_ordinary.policy.update(idx_state, actionProb)
            agent_weighted.policy.update(idx_state, actionProb)
          else:
            actionProb = np.zeros(env.nActions)
            actionProb[env.ACTION_STICK] = 1.0
            agent_ordinary.policy.update(idx_state, actionProb)
            agent_weighted.policy.update(idx_state, actionProb)
    
    print("Experiment : ", idx_experiment)

    errors_ordinary = runExperiment(nEpisodes, env, behaviour_policy, agent_ordinary)
    errors_weighted = runExperiment(nEpisodes, env, behaviour_policy, agent_weighted)
    mse_ordinary = mse_ordinary + (1.0/(idx_experiment+1)) * (errors_ordinary**2 - mse_ordinary)
    mse_weighted = mse_weighted + (1.0/(idx_experiment+1)) * (errors_weighted**2 - mse_weighted)
  
  fig = pl.figure()
  pl.plot(mse_weighted, '-r', label="Weighted IS")
  pl.plot(mse_ordinary, '-g', label="Ordinary IS")
  pl.xlabel("Episodes")
  pl.ylabel("Mean Square Error")
  pl.xscale("log")
  pl.legend()
  pl.show()
  