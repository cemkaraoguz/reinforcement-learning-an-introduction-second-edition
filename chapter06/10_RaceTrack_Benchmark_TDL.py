'''
10_RaceTrack_Benchmark_TDL.py : Benchmark of various TD Learning algorithms on Racetrack (Exercise 5.12)

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import RaceTrack
from IRL.agents.TemporalDifferenceLearning import SARSA, QLearning, ExpectedSARSA, DoubleQLearning
from IRL.utils.Policies import StochasticPolicy
from IRL.utils.Helpers import runSimulation
	
def runExperiment(nEpisodes, env, agent):
	reward_sums = []
	for e in range(nEpisodes):
		
		if(e%100==0):
			print("Episode : ", e)
			
		state = env.reset()
		action = agent.selectAction(state, env.getAvailableActions())
		done = False
		reward_sums.append(0.0)
		while not done:
			
			experiences = [{}]		
			experiences[-1]['state'] = state
			experiences[-1]['action'] = action
			experiences[-1]['done'] = done
			
			new_state, reward, done = env.step(action)
			
			#print("State:", state, "Action: ", env.actionMapping[action][1], "Reward: ", reward, "New state:", new_state)
			
			new_action = agent.selectAction(new_state, env.getAvailableActions(new_state))
			
			xp = {}
			xp['reward'] = reward
			xp['state'] = new_state
			xp['done'] = done
			xp['action'] = new_action
			xp['allowedActions'] = env.getAvailableActions(new_state)
			experiences.append(xp)

			agent.update(experiences)
		
			state = new_state
			
			if(agent.getName()=="SARSA"):
				action = new_action
			else:
				action = agent.selectAction(new_state, env.getAvailableActions(new_state))		
			
			reward_sums[-1] += reward
			
	return reward_sums
	
if __name__=="__main__":

	nExperiments = 5
	nEpisodes = 1000

	# Environment
	trackID = 1
	defaultReward = -1.0
	outOfTrackReward = -1.0
	finishReward = 1.0
	p_actionFail = 0.0

	# Agents
	alpha_SARSA = 0.5
	gamma_SARSA = 1.0	
	alpha_QLearning = 0.5
	gamma_QLearning = 1.0
	alpha_ExpectedSARSA = 0.5
	gamma_ExpectedSARSA = 1.0
	alpha_DoubleQLearning = 0.5
	gamma_DoubleQLearning = 1.0
	
	# Policy
	epsilon_SARSA = 0.01
	epsilon_QLearning = 0.01
	epsilon_ExpectedSARSA = 0.01
	epsilon_DoubleQLearning = 0.01

	if(trackID==0):
		sizeX = 17
		sizeY = 32
		startStates = [(x,31) for x in range(3,9)]
		terminalStates = [(16,y) for y in range(0,6)]
		outOfTrackStates = [(0,0), (1,0), (2,0), (0,1), (1,1), (0,2), (1,2), (0,3)]
		outOfTrackStates.extend([(0	,y) for y in range(14,32)])
		outOfTrackStates.extend([(1	,y) for y in range(22,32)])
		outOfTrackStates.extend([(2	,y) for y in range(29,32)])
		outOfTrackStates.extend([(x	,6) for x in range(10,17)])
		outOfTrackStates.extend([(x	,6) for x in range(10,17)])
		outOfTrackStates.extend([(x	,y) for x in range(9,17) for y in range(7,32)])
	elif(trackID==1):
		sizeX = 32
		sizeY = 30
		startStates = [(x,29) for x in range(0,23)]
		terminalStates = [(31,y) for y in range(0,9)]
		outOfTrackStates = [(x	,0) for x in range(0,16)]
		outOfTrackStates.extend([(x	,1) for x in range(0,13)])
		outOfTrackStates.extend([(x	,2) for x in range(0,12)])
		outOfTrackStates.extend([(x	,y) for x in range(0,11) for y in range(3,7)])
		outOfTrackStates.extend([(x	,7) for x in range(0,12)])
		outOfTrackStates.extend([(x	,8) for x in range(0,13)])
		outOfTrackStates.extend([(x	,y) for x in range(0,14) for y in range(9,14)])
		outOfTrackStates.extend([(x	,14) for x in range(0,13)])
		outOfTrackStates.extend([(x	,15) for x in range(0,12)])
		outOfTrackStates.extend([(x	,16) for x in range(0,11)])
		outOfTrackStates.extend([(x	,17) for x in range(0,10)])
		outOfTrackStates.extend([(x	,18) for x in range(0,9)])
		outOfTrackStates.extend([(x	,19) for x in range(0,8)])
		outOfTrackStates.extend([(x	,20) for x in range(0,7)])
		outOfTrackStates.extend([(x	,21) for x in range(0,6)])
		outOfTrackStates.extend([(x	,22) for x in range(0,5)])
		outOfTrackStates.extend([(x	,23) for x in range(0,4)])
		outOfTrackStates.extend([(x	,24) for x in range(0,3)])
		outOfTrackStates.extend([(x	,25) for x in range(0,2)])
		outOfTrackStates.extend([(x	,26) for x in range(0,1)])
		outOfTrackStates.extend([(x	,9) for x in range(30,32)])
		outOfTrackStates.extend([(x	,10) for x in range(27,32)])
		outOfTrackStates.extend([(x	,11) for x in range(26,32)])
		outOfTrackStates.extend([(x	,12) for x in range(24,32)])
		outOfTrackStates.extend([(x	,y) for x in range(23,32) for y in range(13,30)])
	else:
		sys.exit("ERROR: trackID not recognized")
		
	avg_reward_sums_SARSA = np.zeros(nEpisodes)
	avg_reward_sums_QLearning = np.zeros(nEpisodes)
	avg_reward_sums_ExpectedSARSA = np.zeros(nEpisodes)
	avg_reward_sums_DoubleQLearning = np.zeros(nEpisodes)
	for idx_experiment in range(nExperiments):
	
		print("Experiment : ", idx_experiment)
			
		env = RaceTrack(sizeX, sizeY, startStates=startStates, terminalStates=terminalStates, impassableStates=outOfTrackStates,
						defaultReward=defaultReward, crashReward=outOfTrackReward, finishReward=finishReward, p_actionFail=p_actionFail)
		
		agent_SARSA = SARSA(env.nStates, env.nActions, alpha_SARSA, gamma_SARSA, epsilon=epsilon_SARSA)
		agent_QLearning = QLearning(env.nStates, env.nActions, alpha_QLearning, gamma_QLearning, epsilon=epsilon_QLearning)
		agent_ExpectedSARSA = ExpectedSARSA(env.nStates, env.nActions, alpha_ExpectedSARSA, gamma_ExpectedSARSA, epsilon=epsilon_ExpectedSARSA)
		agent_DoubleQLearning = DoubleQLearning(env.nStates, env.nActions, alpha_DoubleQLearning, gamma_DoubleQLearning, epsilon=epsilon_DoubleQLearning)
		
    reward_sums_SARSA = runExperiment(nEpisodes, env, agent_SARSA)
		reward_sums_QLearning = runExperiment(nEpisodes, env, agent_QLearning)
		reward_sums_ExpectedSARSA = runExperiment(nEpisodes, env, agent_ExpectedSARSA)
		reward_sums_DoubleQLearning = runExperiment(nEpisodes, env, agent_DoubleQLearning)
		
		avg_reward_sums_SARSA = avg_reward_sums_SARSA + (1.0/(idx_experiment+1))*(reward_sums_SARSA - avg_reward_sums_SARSA)
		avg_reward_sums_QLearning = avg_reward_sums_QLearning + (1.0/(idx_experiment+1))*(reward_sums_QLearning - avg_reward_sums_QLearning)
		avg_reward_sums_ExpectedSARSA = avg_reward_sums_ExpectedSARSA + (1.0/(idx_experiment+1))*(reward_sums_ExpectedSARSA - avg_reward_sums_ExpectedSARSA)
		avg_reward_sums_DoubleQLearning = avg_reward_sums_DoubleQLearning + (1.0/(idx_experiment+1))*(reward_sums_DoubleQLearning - avg_reward_sums_DoubleQLearning)
	
	pl.figure()
	pl.plot(avg_reward_sums_SARSA, '-b', label='SARSA')
	pl.plot(avg_reward_sums_QLearning, '-r', label='Q-Learning')
	pl.plot(avg_reward_sums_ExpectedSARSA, '-g', label='Expected SARSA')
	pl.plot(avg_reward_sums_DoubleQLearning, '-k', label='Double Q-Learning')
	pl.xlabel("Episodes")
	pl.ylabel("Sum of reward during episodes")
	pl.legend()	
	pl.show()
		
	agents = [agent_SARSA, agent_QLearning, agent_ExpectedSARSA, agent_DoubleQLearning]
	for agent in agents:
		print("Policy for :", agent.getName())
		env.printEnv(agent)
	
	for agent in agents:
		input("Press any key to simulate agent "+agent.getName())
		agentHistory = runSimulation(env, agent)
		print("Simulation:", agent.getName())	
		env.render(agentHistory)