'''
06_MaximizationBias.py : replication of figure 6.5

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from IRL.environments.ToyExamples import MaximizationBias
from IRL.agents.TemporalDifferenceLearning import QLearning, DoubleQLearning
from IRL.utils.Policies import ActionValuePolicy

def runExperiment(nEpisodes, env, agent):
	countLeftFromA = np.zeros(nEpisodes)
	for e in range(nEpisodes):
		
		if(e%10==0):
			print("Episode : ", e)
		
		state = env.reset()
		done = False
		while not done:
			
			experiences = [{}]
			
			action = agent.selectAction(state, env.getAvailableActions())
			
			experiences[-1]['state'] = state
			experiences[-1]['action'] = action
			experiences[-1]['done'] = done
			
			if((state==env.STATE_A) and (action==env.ACTION_LEFT)):
				countLeftFromA[e] += 1
			
			new_state, reward, done = env.step(action)
			
			#print(agent.getName(), "Episode:", e, "State:", state, "Action: ", action, "Reward: ", reward, "New state:", new_state)

			xp = {}
			xp['reward'] = reward
			xp['state'] = new_state
			xp['done'] = done
			xp['allowedActions'] = env.getAvailableActions(new_state)
			experiences.append(xp)
			
			agent.update(experiences)
		
			state = new_state
	
	return countLeftFromA

if __name__=="__main__":

	nExperiments = 10000
	nEpisodes = 300
	
	# Agents
	alpha_QLearning = 0.1
	gamma_QLearning = 1.0
	alpha_DoubleQLearning = 0.1
	gamma_DoubleQLearning = 1.0
	
	# Policy
	epsilon_QLearning = 0.1
	epsilon_DoubleQLearning = 0.1
	
	# Environment
	env = MaximizationBias()
	
	#env.printEnv()
	
	allCountLeftFromA_QLearning = np.zeros(nEpisodes)
	allCountLeftFromA_DoubleQLearning = np.zeros(nEpisodes)
	for idx_experiment in range(nExperiments):

		print("Experiment : ", idx_experiment)
		
		agent_QLearning = QLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, epsilon=epsilon_QLearning)
		agent_DoubleQLearning = DoubleQLearning(env.nStates, env.nActions, alpha_DoubleQLearning, gamma_DoubleQLearning, epsilon=epsilon_DoubleQLearning)
		countLeftFromA_QLearning = runExperiment(nEpisodes, env, agent_QLearning)
		countLeftFromA_DoubleQLearning = runExperiment(nEpisodes, env, agent_DoubleQLearning)
		allCountLeftFromA_QLearning += countLeftFromA_QLearning
		allCountLeftFromA_DoubleQLearning += countLeftFromA_DoubleQLearning
		
	fig = pl.figure()
	pl.plot(allCountLeftFromA_QLearning/nExperiments*100, '-r', label="Q-Learning")
	pl.plot(allCountLeftFromA_DoubleQLearning/nExperiments*100, '-g', label="Double Q-Learning")
	pl.plot(np.ones(nEpisodes)*5.0, '--k')
	pl.xlabel("Episodes")
	pl.ylabel("% left actions from A")
	pl.legend()
	pl.show()
	