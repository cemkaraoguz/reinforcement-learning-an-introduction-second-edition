'''
Helpers.py : Some helper functions

Cem Karaoguz, 2020
MIT License
'''
from time import sleep

def getValueFromDict(indict, key, defaultVal=None):
  if key in indict.keys():
    return indict[key]
  else:
    return defaultVal

def runSimulation(env, agent, t_max=10000): 
  state = env.reset()
  done = False
  agentHistory = []
  t = 0
  while not done:
    agentHistory.append(state)
    env.render()
    action = agent.getGreedyAction(state, env.getAvailableActions())
    state, reward, done = env.step(action)
    print("t:", t, "state : ", state, " action : ", action, " reward : ", reward )
    sleep(0.05)
    t+=1
    if t>t_max:
      break
  return agentHistory