'''
09_Blackjack_Benchmark_TDL.py : Benchmark of various TD Learning algorithms on Blackjack (Example 5.3)

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from IRL.environments.Gambler import Blackjack
from IRL.agents.TemporalDifferenceLearning import SARSA, QLearning, ExpectedSARSA, DoubleQLearning
from IRL.utils.Policies import StochasticPolicy
  
def runExperiment(nEpisodes, env, agent):
  reward_sums = []
  for e in range(nEpisodes):
    
    if(e%10000==0):
      print("Episode : ", e)
      
    state = env.reset()
    action = agent.selectAction(state)
    done = False
    while not done:

      experiences = [{}]
      experiences[-1]['state'] = state
      experiences[-1]['action'] = action
      experiences[-1]['done'] = done      
      
      new_state, reward, done = env.step(action)
      
      #print("State:", state, "Action: ", env.actionMapping[action][1], "Reward: ", reward, "New state:", new_state)

      new_action = agent.selectAction(new_state)    
      
      xp = {}
      xp['reward'] = reward
      xp['state'] = new_state
      xp['done'] = done
      xp['action'] = new_action
      experiences.append(xp)

      agent.update(experiences)
    
      state = new_state
      
      if(agent.getName()=="SARSA"):
        action = new_action
      else:
        action = agent.selectAction(new_state)

if __name__=="__main__":

  nExperiments = 1
  nEpisodes = 200000

  # Environment

  # Agents
  alpha_SARSA = 0.1
  gamma_SARSA = 1.0
  alpha_QLearning = 0.1
  gamma_QLearning = 1.0
  alpha_ExpectedSARSA = 0.1
  gamma_ExpectedSARSA = 1.0
  alpha_DoubleQLearning = 0.1
  gamma_DoubleQLearning = 1.0
  
  # Policy
  epsilon_SARSA = 0.2
  epsilon_QLearning = 0.2
  epsilon_ExpectedSARSA = 0.2
  epsilon_DoubleQLearning = 0.2
  
  for idx_experiment in range(nExperiments):
  
    print("Experiment : ", idx_experiment)
      
    env = Blackjack()
    
    agent_SARSA = SARSA(env.nStates, env.nActions, alpha_SARSA, gamma_SARSA, epsilon=epsilon_SARSA)
    agent_QLearning = QLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, epsilon=epsilon_QLearning)
    agent_ExpectedSARSA = ExpectedSARSA(env.nStates, env.nActions, alpha_ExpectedSARSA, gamma_ExpectedSARSA, epsilon=epsilon_ExpectedSARSA)
    agent_DoubleQLearning = DoubleQLearning(env.nStates, env.nActions, alpha_DoubleQLearning, gamma_DoubleQLearning, epsilon=epsilon_DoubleQLearning)
        
    runExperiment(nEpisodes, env, agent_SARSA)
    runExperiment(nEpisodes, env, agent_QLearning)
    runExperiment(nEpisodes, env, agent_ExpectedSARSA)
    runExperiment(nEpisodes, env, agent_DoubleQLearning)

  agents = [agent_SARSA, agent_QLearning, agent_ExpectedSARSA, agent_DoubleQLearning]
  for agent in agents:
    value_usableace = np.zeros([env.nStatesDealerShowing, env.nStatesPlayerSum])
    value_nousableace = np.zeros([env.nStatesDealerShowing, env.nStatesPlayerSum])
    print_str_usableace = ""
    print_str_nousableace = ""
    for i in range(env.nStatesPlayerSum-1, -1, -1):
      for j in range(env.nStatesDealerShowing):
        idx_usableace = env.getLinearIndex(env.minPlayerSum+i, env.minDealerShowing+j, env.USABLE_ACE_YES)
        idx_nousableace = env.getLinearIndex(env.minPlayerSum+i, env.minDealerShowing+j, env.USABLE_ACE_NO)
        action_usableace = agent.getGreedyAction(idx_usableace)
        action_nousableace = agent.getGreedyAction(idx_nousableace)
        value_usableace[j,i] = agent.getValue(idx_usableace)
        value_nousableace[j,i] = agent.getValue(idx_nousableace)
        print_str_usableace += str(env.actionMapping[action_usableace][1]) + "\t"
        print_str_nousableace += str(env.actionMapping[action_nousableace][1]) + "\t"
      print_str_usableace += "\n"
      print_str_nousableace += "\n"
    print(agent.getName(), "Policy (usable Ace)")
    print(print_str_usableace)
    print() 
    print(agent.getName(), "Policy (No usable Ace)")
    print(print_str_nousableace)
    
    X = np.arange(0, env.nStatesPlayerSum, 1)+12
    Y = np.arange(0, env.nStatesDealerShowing, 1)
    Y, X = np.meshgrid(Y, X)
    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, np.transpose(value_usableace), linewidth=0, antialiased=False)
    ax.set_ylabel("Dealer showing")
    ax.set_xlabel("Player sum")
    ax.set_title(agent.getName()+" - Usable Ace")

    fig = pl.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, np.transpose(value_nousableace), linewidth=0, antialiased=False)
    ax.set_ylabel("Dealer showing")
    ax.set_xlabel("Player sum")
    ax.set_title(agent.getName()+" - No usable Ace")
    
  pl.show()