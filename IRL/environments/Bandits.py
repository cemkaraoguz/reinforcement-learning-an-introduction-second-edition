'''
Bandits.py : Bandit problems

Cem Karaoguz, 2020
MIT License
'''

import numpy as np

class KArmedBandit:

  def __init__(self, nBandits, rewardDist_mean=0.0, rewardDist_std=1.0):
    self.nStates = 1
    self.nActions = nBandits
    self.rewardDist_mean = rewardDist_mean
    self.rewardDist_std = rewardDist_std
    self.rewardMeans = np.random.normal(self.rewardDist_mean, self.rewardDist_std, self.nActions)
    
  def step(self, action):
    return np.random.normal(self.rewardMeans[action], self.rewardDist_std)
    
  def addGaussianNoiseToRewards(self, noise_mean, noise_std):
    self.rewardMeans = self.rewardMeans + np.random.normal(noise_mean, noise_std, self.nActions) 
    
  def reset(self):
    self.rewardMeans = np.random.normal(self.rewardDist_mean, self.rewardDist_std, self.nActions)