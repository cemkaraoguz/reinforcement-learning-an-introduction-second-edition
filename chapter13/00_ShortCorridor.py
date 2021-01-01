'''
00_ShortCorridor.py : Replication of figures 13.1, 13.2

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.ToyExamples import ShortCorridor
from IRL.agents.PolicyGradient import REINFORCE, REINFORCEwithBaseline
from IRL.utils.ApproximationFunctions import linearTransform, dLinearTransform, softmaxLinear, dLogSoftmaxLinear
from IRL.utils.FeatureTransformations import FixedStateEncoding, tileCoding

def runExperiment(nEpisodes, env, agent, nStepsMax):
  nStepsPerEpisode = np.zeros(nEpisodes)
  rewards = np.zeros(nEpisodes)
  for e in range(nEpisodes):
    
    if e%10==0:
      print("Episode : ", e)
    
    experiences = [{}]
    done = False
    state = env.reset()
    t = 0
    cum_reward = 0.0

    while not done:     

      action = agent.selectAction(state)

      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done

      new_state, reward, done = env.step(action)
      
      #print("Episode:", e, "State:", state, "Action: ", action, "Reward: ", reward, "New state:", new_state, "done:", done)
      
      xp = {}
      xp['reward'] = reward
      xp['state'] = new_state
      xp['done'] = done
      experiences.append(xp)
      
      state = new_state
      cum_reward+=reward
      t += 1
      if t>=nStepsMax:
        print("oh no!")
        break
    
    agent.update(experiences)
      
    nStepsPerEpisode[e] = t
    rewards[e] = cum_reward
  
  return nStepsPerEpisode, rewards
  
if __name__=="__main__":
  nExperiments = 100
  nEpisodes = 1000
  maxSteps = 1000
  v_star = -11.6
  
  # Parameters for REINFORCE
  alphas_REINFORCE =  [2**(-12), 2**(-13), 2**(-14)]
  gamma = 1.0

  # Parameters for REINFORCE with baseline
  alpha_theta = 2**(-9)
  alpha_w = 2**(-6)
   
  env = ShortCorridor()
  
  # Function approximation parameters for policies
  nParams_theta = 2
  stateActionEncodingMatrix = np.zeros([env.nStates, env.nActions, nParams_theta])
  stateActionEncodingMatrix[:,env.ACTION_LEFT,:] = np.array([0,1])
  stateActionEncodingMatrix[:,env.ACTION_RIGHT,:] = np.array([1,0])
  policyApproximationFunctionArgs = {'af':softmaxLinear, 'afd':dLogSoftmaxLinear, 
      'ftf':FixedStateEncoding, 'stateEncodingMatrix':stateActionEncodingMatrix, 'nActions':env.nActions}

  # Function approximation parameter for value function
  nParams_w = 1
  stateEncodingMatrix = np.ones([env.nStates, nParams_w])
  approximationFunctionArgs = {'af':linearTransform, 'afd':dLinearTransform, 'ftf':FixedStateEncoding, 'stateEncodingMatrix':stateEncodingMatrix}
  
  agent_REINFORCE = REINFORCE(0.0, gamma, nParams_theta, env.nActions, policyApproximationFunctionArgs)
  agent_REINFORCEwB = REINFORCEwithBaseline(alpha_w, alpha_theta, gamma, nParams_w, approximationFunctionArgs, 
    nParams_theta, env.nActions, policyApproximationFunctionArgs)
  
  rewards_avg_REINFORCE = []
  for i, alpha in enumerate(alphas_REINFORCE):
    rewards_avg_REINFORCE.append( np.zeros(nEpisodes) )
    for idx_experiment in range(1,nExperiments+1):
      print("Experiment:", idx_experiment)
      agent_REINFORCE.reset()
      agent_REINFORCE.alpha = alpha
      nStepsPerEpisode_REINFORCE, rewards_REINFORCE = runExperiment(nEpisodes, env, agent_REINFORCE, maxSteps)
      rewards_avg_REINFORCE[i] = rewards_avg_REINFORCE[i] + (1.0/idx_experiment)*(rewards_REINFORCE - rewards_avg_REINFORCE[i])
  
  rewards_avg_REINFORCEwB = np.zeros(nEpisodes)
  for idx_experiment in range(1,nExperiments+1):
    print("Experiment:", idx_experiment)
    agent_REINFORCEwB.reset()
    nStepsPerEpisode_REINFORCEwB, rewards_REINFORCEwB = runExperiment(nEpisodes, env, agent_REINFORCEwB, maxSteps)
    rewards_avg_REINFORCEwB = rewards_avg_REINFORCEwB + (1.0/idx_experiment)*(rewards_REINFORCEwB - rewards_avg_REINFORCEwB)
      
  #---------------------
  #    Visualisation 
  #---------------------
  pl.figure()
  for i in range(len(rewards_avg_REINFORCE)):
    pl.plot(rewards_avg_REINFORCE[i], label="REINFORCE, alpha="+str(alphas_REINFORCE[i]))
  pl.plot(rewards_avg_REINFORCEwB, label="REINFORCE with Baseline, alpha_w="+str(alpha_w)+", alpha_theta="+str(alpha_theta))
  pl.plot(np.zeros(nEpisodes)+v_star,'--k')
  pl.legend()
  pl.xlabel("Episodes")
  pl.ylabel("Total reward on episode averaged over "+str(nExperiments)+" runs")
  pl.show()