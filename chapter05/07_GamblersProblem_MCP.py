'''
07_GamblersProblem_MCP.py : Application of a Monte Carlo solution to Gambler's Problem (prediction)

Cem Karaoguz, 2020
MIT License
'''

import numpy as np
import pylab as pl

from IRL.environments.Gambler import CoinFlipGame
from IRL.agents.MonteCarlo import MonteCarloPrediction
from IRL.utils.Policies import StochasticPolicy

if __name__=="__main__":

	nEpisodes = 100000

	# Environment
	maxCapital = 100
	prob_heads = 0.4

	# Agent
	gamma = 1.0

	env = CoinFlipGame(maxCapital, prob_heads)
	policy = StochasticPolicy(env.nStates, env.nActions)
	agent = MonteCarloPrediction(env.nStates, gamma, doUseAllVisits=False)
	
	#env.printEnv()
	
	for e in range(nEpisodes):
	
		if(e%1000==0):
			print("Episode : ", e)
			
		experiences = [{}]
		state = env.reset()
		done = False	
		while not done:
		
			action = policy.sampleAction(state, env.getAvailableActions())
			
			experiences[-1]['state'] = state
			experiences[-1]['action'] = action
			experiences[-1]['done'] = done
			
			new_state, reward, done = env.step(action)

			#print("Episode : ", e, " State : ", state, " Action : ", action, " Reward : ", reward, " Next state : ", new_state)
			
			xp = {}
			xp['reward'] = reward
			xp['state'] = new_state
			xp['done'] = done
			experiences.append(xp)
			
			state = new_state
      
		agent.evaluate(experiences)
	
	pl.figure()
	pl.plot(agent.valueTable[1:-1])
  pl.xlabel("Capital")
  pl.ylabel("Value estimates")
	pl.show()