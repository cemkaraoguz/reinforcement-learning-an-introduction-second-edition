'''
01_BairdsCounterExample.py : Replication of figures 11.2, 11.5 and 11.6

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.ToyExamples import BairdsCounterExample
from IRL.agents.TemporalDifferenceApproximation import SemiGradientOffPolicyTDPrediction, SemiGradientQLearningTDPrediction, GradientTDPrediction
from IRL.agents.DynamicProgramming import SemiGradientPolicyEvaluation, ExpectedTDC, EmphaticTDPolicyEvaluation
from IRL.utils.Policies import StochasticPolicy
from IRL.utils.ApproximationFunctions import linearTransform, dLinearTransform
from IRL.utils.FeatureTransformations import FixedStateEncoding
from IRL.utils.Helpers import getValueFromDict

def calculateProjectionMatrix(nStates, nParams, ftf_args, mu):
  ftf = getValueFromDict(ftf_args, "ftf")
  X = np.zeros((nStates, nParams))
  for s in range(nStates):
    X[s, :] = ftf(s, **ftf_args)
  D = np.diag(mu)
  return X @ np.linalg.pinv(X.T @ D @ X) @ X.T @ D
  
def calculatePBE(env, agent, mu, policy):
  pm = calculateProjectionMatrix(env.nStates, agent.nParams, agent.af_kwargs, mu)
  bellman_error = np.zeros(env.nStates)
  for s in range(env.nStates):
    aux = [policy.getProbability(s,a) * env.computeExpectedValue(s, a, agent.w, agent.af_kwargs, agent.gamma) for a in range(env.nActions)]
    bellman_error[s] = np.sum(aux) - agent.getValue(s)
  return np.dot(pm, bellman_error)
  
def runExperiment(env, agent, behaviour_policy, v_targetPolicy=None):
  experiences = [{}]
  done = False
  state = env.reset()
  trace_weights = []
  trace_VE = []
  trace_PBE = []
  mu = np.zeros(env.nStates)
  for t in range(maxSteps):
    
    action = behaviour_policy.sampleAction(state)

    experiences[-1]['state'] = state
    experiences[-1]['action'] = action
    experiences[-1]['done'] = done
    mu[state] += 1
    
    new_state, reward, done = env.step(action)
    
    print(agent.getName(), "Step:", t, "State:", state, "Action: ", env.actionMapping[action][1], 
          "Reward: ", reward, "New state:", new_state, "done:", done)
    
    xp = {}
    xp['reward'] = reward
    xp['state'] = new_state
    xp['done'] = done
    experiences.append(xp)
    
    agent.update(experiences, behaviour_policy)
    
    state = new_state
    
    trace_weights.append(np.array(agent.w))
    if v_targetPolicy is not None:
      v_curr = np.array([agent.getValue(s) for s in range(env.nStates)], dtype=float)
      trace_VE.append( np.sqrt(np.sum((mu/(t+1))*(v_targetPolicy - v_curr)**2)) )
      trace_PBE.append( np.sqrt(np.sum(calculatePBE(env, agent, mu/(t+1), agent.policy)**2)) )
    
  return trace_weights, trace_VE, trace_PBE
  
def visualizeResults(agent, trace_weights, trace_VE=None, trace_PBE=None):
  pl.figure()
  for i in range(agent.nParams):
    pl.plot(np.array(trace_weights)[:,i], label="w"+str(i+1))
  if trace_VE is not None:
    pl.plot(trace_VE, label="sqrt(VE)")
  if trace_PBE is not None:
    pl.plot(trace_PBE, label="sqrt(PBE)")
  pl.plot(np.zeros(maxSteps),"--k")
  pl.xlabel("Steps")
  pl.title(agent.getName())
  pl.legend()
  
if __name__=="__main__":
  maxSteps = 1000
  nParams = 8
  alpha_TD = 0.01 
  gamma_TD = 0.99
  initialWeights = np.array([1, 1, 1, 1, 1, 1, 10, 1], dtype=float)
  
  alpha_DP = 0.01 
  gamma_DP = 0.99
  thresh_convergence = 1e-10
  
  alpha_QL = 0.01
  gamma_QL = 0.99

  alpha_GTD = 0.005 
  beta_GTD = 0.05
  gamma_GTD = 0.99
  
  alpha_ETD = 0.03
  gamma_ETD = 0.99
  
  env = BairdsCounterExample()
  behaviour_policy = StochasticPolicy(env.nStates, env.nActions)
  behaviour_policy.actionProbabilityTable[:,env.ACTION_IDX_DASHED] = 6.0/7.0
  behaviour_policy.actionProbabilityTable[:,env.ACTION_IDX_SOLID] = 1.0/7.0

  target_policy = StochasticPolicy(env.nStates, env.nActions)
  target_policy.actionProbabilityTable[:,:] = 0.0
  target_policy.actionProbabilityTable[:,env.ACTION_IDX_SOLID] = 1.0
  
  stateEncodingMatrix = np.zeros([env.nStates, nParams])
  for i in range(env.nStates-1):
    stateEncodingMatrix[i,i] = 2
    stateEncodingMatrix[i,7] = 1
  stateEncodingMatrix[6,6] = 1
  stateEncodingMatrix[6,7] = 2
  approximationFunctionArgs = {'af':linearTransform, 'afd':dLinearTransform, 'ftf':FixedStateEncoding, 'stateEncodingMatrix':stateEncodingMatrix}

  w_targetPolicy = np.array([1, 1, 1, 1, 1, 1, 4, -2], dtype=float)
  v_targetPolicy = np.array([linearTransform(w_targetPolicy, state, **approximationFunctionArgs) for state in range(env.nStates)], dtype=float)
  
  #---------------------
  #     DP
  #---------------------
  agent_DP = SemiGradientPolicyEvaluation(env.nStates, env.nActions, nParams, gamma_DP, alpha_DP, thresh_convergence, 
    expectedValueFunction=env.computeExpectedValue, approximationFunctionArgs=approximationFunctionArgs)
  agent_DP.w[:] = initialWeights
  trace_weights_DP = []
  for t in range(maxSteps):
    
    print(agent_DP.getName(), " step:", t)
    
    deltaMax, isConverged = agent_DP.evaluate(target_policy)

    trace_weights_DP.append(np.array(agent_DP.w))

  #---------------------
  #     TD 
  #---------------------
  agent_TD = SemiGradientOffPolicyTDPrediction(nParams, alpha_TD, gamma_TD, target_policy, approximationFunctionArgs)
  agent_TD.w[:] = initialWeights
  trace_weights_TD, _, _ = runExperiment(env, agent_TD, behaviour_policy)
  
  #---------------------
  #     QL 
  #---------------------
  agent_QL = SemiGradientQLearningTDPrediction(nParams, env.nActions, alpha_QL, gamma_QL, target_policy, 
    approximationFunctionArgs=approximationFunctionArgs)
  agent_QL.w[:] = initialWeights
  trace_weights_QL, _, _ = runExperiment(env, agent_QL, behaviour_policy)
  
  #---------------------
  #     GTD 
  #---------------------
  agent_GTD = GradientTDPrediction(nParams, alpha_GTD, beta_GTD, gamma_GTD, target_policy, approximationFunctionArgs)
  agent_GTD.w[:] = initialWeights
  trace_weights_GTD, trace_VE_GTD, trace_PBE_GTD = runExperiment(env, agent_GTD, behaviour_policy, v_targetPolicy=v_targetPolicy)
  
  #---------------------
  #     EGTD
  #---------------------
  agent_EGTD = ExpectedTDC(env.nStates, env.nActions, nParams, gamma_GTD, alpha_GTD, beta_GTD, env.defaultReward, 
    approximationFunctionArgs=approximationFunctionArgs)
  agent_EGTD.w[:] = initialWeights
  trace_weights_EGTD = []
  trace_VE_EGTD = []
  trace_PBE_EGTD = []
  mu_EGTD = np.zeros(env.nStates)
  for t in range(maxSteps):
    
    print(agent_EGTD.getName(), " step:", t)
    
    agent_EGTD.evaluate(target_policy, behaviour_policy)
    mu_EGTD += 1

    trace_weights_EGTD.append(np.array(agent_EGTD.w))
    
    v_curr = np.array([agent_EGTD.getValue(s) for s in range(env.nStates)], dtype=float)
    trace_VE_EGTD.append( np.sqrt(np.sum((mu_EGTD/(t+1))*(v_targetPolicy - v_curr)**2)) )
    trace_PBE_EGTD.append( np.sqrt(np.sum(calculatePBE(env, agent_EGTD, mu_EGTD/(t+1), target_policy)**2)) )

  #---------------------
  #     ETD 
  #---------------------
  agent_ETD = EmphaticTDPolicyEvaluation(env.nStates, env.nActions, nParams, gamma_ETD, alpha_ETD, thresh_convergence,
    expectedValueFunction=env.computeExpectedValue, approximationFunctionArgs=approximationFunctionArgs)
  agent_ETD.w[:] = initialWeights
  trace_weights_ETD = []
  trace_VE_ETD = []
  mu_ETD = np.zeros(env.nStates)
  for t in range(maxSteps):
    
    print(agent_ETD.getName(), " step:", t)
    
    deltaMax, isConverged = agent_ETD.evaluate(behaviour_policy)
    mu_ETD += 1

    trace_weights_ETD.append(np.array(agent_ETD.w))
    
    v_curr = np.array([agent_ETD.getValue(s) for s in range(env.nStates)], dtype=float)
    trace_VE_ETD.append( np.sqrt(np.sum((mu_ETD/(t+1))*(v_targetPolicy - v_curr)**2)) )

  #---------------------
  #    Visualisation 
  #---------------------
  visualizeResults(agent_TD, trace_weights_TD, trace_VE=None, trace_PBE=None)
  visualizeResults(agent_DP, trace_weights_DP, trace_VE=None, trace_PBE=None)
  visualizeResults(agent_QL, trace_weights_QL, trace_VE=None, trace_PBE=None)
  visualizeResults(agent_GTD, trace_weights_GTD, trace_VE=trace_VE_GTD, trace_PBE=trace_PBE_GTD)
  visualizeResults(agent_EGTD, trace_weights_EGTD, trace_VE=trace_VE_EGTD, trace_PBE=trace_PBE_EGTD)
  visualizeResults(agent_ETD, trace_weights_ETD, trace_VE=trace_VE_ETD, trace_PBE=None)
  pl.show() 