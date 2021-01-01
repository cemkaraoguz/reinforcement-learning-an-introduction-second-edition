'''
00_GridWorld_1_PE.py : replication of Figure 3.2

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gridworlds import DeterministicGridWorld
from IRL.agents.DynamicProgramming import PolicyEvaluation
from IRL.utils.Policies import StochasticPolicy

if __name__=="__main__":

	nEpisodes = 1000

	# Environment
	sizeX = 5
	sizeY = 5
	defaultReward = 0.0
	outOfGridReward = -1.0
	specialRewards = {((1,0),0):10,((1,0),1):10,((1,0),2):10,((1,0),3):10, 
    ((3,0),0):5, ((3,0),1):5, ((3,0),2):5, ((3,0),3):5}
	specialStateTransitions = {(1,0):(1,4), (3,0):(3,2)}

	# Agent
	gamma = 0.9
	thresh_convergence = 1e-30

	env = DeterministicGridWorld(sizeX, sizeY, specialRewards=specialRewards, specialStateTransitions=specialStateTransitions, 
    defaultReward=defaultReward, outOfGridReward=outOfGridReward)
	policy = StochasticPolicy(env.nStates, env.nActions)
	agent = PolicyEvaluation(env.nStates, env.nActions, gamma, thresh_convergence, env.computeExpectedValue)
	
	env.printEnv()
	
	for e in range(nEpisodes):
			
		deltaMax, isConverged = agent.evaluate(policy)
		
		print("Episode : ", e, " Delta: ", deltaMax)
		
		printStr = ""
		for y in range(sizeY):
			for x in range(sizeX):
				i = env.getLinearIndex(x,y)
				printStr += "{:.2f}".format(agent.valueTable[i]) + "\t"
			printStr += "\n"
			
		print(printStr)
		
		if(isConverged):
			print("Convergence achieved!")
			break
