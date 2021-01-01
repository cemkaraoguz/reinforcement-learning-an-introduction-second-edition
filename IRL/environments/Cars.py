import numpy as np

class MountainCar:

  # Velocity dynamics
  A = 0.001
  B = 0.0025
  
  def __init__(self, positionBounds, velocityBounds, startPositionBounds, defaultReward=-1.0, terminalReward=None):
    self.minX = positionBounds[0]
    self.maxX = positionBounds[1]
    self.minXd = velocityBounds[0]
    self.maxXd = velocityBounds[1]
    self.minStartX = startPositionBounds[0]
    self.maxStartX = startPositionBounds[1]
    self.defaultReward = defaultReward
    if terminalReward is None:
      self.terminalReward = defaultReward
    else:
      self.terminalReward = terminalReward
    assert self.minX<self.maxX
    assert self.minXd<self.maxXd
    assert self.minStartX<self.maxStartX
    self.actionMapping = {0:(-1, "-1"), 1:(0, " 0"), 2:(1, "+1")}
    self.nActions = len(self.actionMapping)
    self.X = None
    self.Xd = None
    self.reset()
  
  def step(self, action):
    reward = self.defaultReward
    done = False
    # Calculate velocity
    newVel = self.Xd + self.A*self.actionMapping[action][0] - self.B*np.cos(3.0*self.X)
    newVel = min(newVel, self.maxXd)
    newVel = max(newVel, self.minXd)
    self.Xd = newVel
    # Calculate position
    newPos = self.X + self.Xd
    newPos = min(newPos, self.maxX)
    newPos = max(newPos, self.minX)
    self.X = newPos
    # Check conditions on position
    if self.X<=self.minX:
      self.xD = 0.0
    if self.X>=self.maxX:
      reward = self.terminalReward
      done = True
    return [self.X, self.Xd], reward, done
  
  def reset(self):
    self.X = np.random.uniform(low=self.minStartX, high=self.maxStartX)
    self.Xd = 0.0
    return [self.X, self.Xd]
    
  def render(self):
    pass
    
  def getAvailableActions(self):
    return np.array(range(self.nActions))