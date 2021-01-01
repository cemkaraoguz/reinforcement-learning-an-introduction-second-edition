'''
Gambler.py : Implementations of gambler problems

Cem Karaoguz, 2020
MIT License
'''

import numpy as np

class CoinFlipGame():

  def __init__(self, maxCapital, prob_heads, reward_default=0.0, reward_win=1.0):
    self.maxCapital = maxCapital
    self.prob_heads = prob_heads
    self.reward_default = reward_default
    self.reward_win = reward_win
    self.nStates = self.maxCapital+1
    self.nActions = self.maxCapital+1
    self.terminalStates = [0, self.maxCapital]
    self.actionsAllowed = np.zeros([self.nStates, self.nActions], dtype=np.int)
    for idx_state  in range(1,self.nStates):
      self.actionsAllowed[idx_state,1:idx_state+1] = 1
    self.actionsAllowed[0,0] = 1
    self.currentCapital = np.random.randint(1, self.maxCapital-1)
  
  def computeExpectedValue(self, idx_state, idx_action, valueTable, gamma): 
    if(idx_state<=0):
      return self.reward_default
    if(idx_state>=self.maxCapital):
      return self.reward_default
    # Win
    idx_state_win = idx_state + idx_action
    if(idx_state_win>=self.maxCapital):
      reward_win = self.reward_win
      idx_state_win = self.maxCapital
    else:
      reward_win = self.reward_default
    expectedValue_win = self.prob_heads * (reward_win + gamma * valueTable[idx_state_win])
    # Lose
    idx_state_lose = idx_state - idx_action
    reward_lose = self.reward_default
    expectedValue_lose = (1.0 - self.prob_heads) * (reward_lose + gamma * valueTable[idx_state_lose])
    return (expectedValue_win + expectedValue_lose)
  
  def step(self, action):
    done = False
    coin_toss = np.random.binomial(1, self.prob_heads)
    if(coin_toss==1):
      # Win
      self.currentCapital = min(self.currentCapital + (action), self.maxCapital)
    else:
      # Lose
      self.currentCapital = max(self.currentCapital - (action), 0)
    if(self.currentCapital>=self.maxCapital):
      reward = self.reward_win
    else:
      reward = self.reward_default
    if(self.currentCapital in self.terminalStates):
      done = True
    return self.currentCapital, reward, done
    
  def reset(self):
    self.currentCapital = np.random.randint(1, self.maxCapital-1)
    return self.currentCapital
    
  def getAllowedActionsMask(self, idx=None):
    i = idx if idx is not None else self.currentCapital
    return self.actionsAllowed[i, :]

  def getAvailableActions(self, i=None):
    return np.nonzero(self.getAllowedActionsMask(i))[0]
    
class Blackjack():
  
  IDX_DEALER_CARD_SHOWN = 0
  LABEL_ACE = 1
  USABLE_ACE_YES = 1
  USABLE_ACE_NO = 0
  VAL_USABLE_ACE = 11
  VAL_NONUSABLE_ACE = 1
  VAL_FACECARDS = 10
  VAL_BLACKJACK = 21 
  ACTION_HIT = 0
  ACTION_STICK = 1
  N_CARDS_DECK = 13
  
  def __init__(self):
    # Rewards
    self.reward_default = 0.0
    self.reward_win = 1.0
    self.reward_lose = -1.0
    self.reward_draw = 0.0
    # States
    self.minPlayerSum = 12
    self.maxPlayerSum = self.VAL_BLACKJACK
    self.nStatesPlayerSum = self.maxPlayerSum - self.minPlayerSum + 1
    self.minDealerShowing = self.LABEL_ACE
    self.maxDealerShowing = self.VAL_FACECARDS
    self.nStatesDealerShowing = self.maxDealerShowing - self.minDealerShowing + 1
    self.nStatesPlayerHasUsableAce = 2
    self.nStates = self.nStatesPlayerSum * self.nStatesDealerShowing * self.nStatesPlayerHasUsableAce + 3
    self.state_win = self.nStates - 1
    self.state_draw = self.nStates - 2
    self.state_lose = self.nStates - 3
    # Actions
    self.actionMapping = {0:(self.ACTION_HIT, "Hit"), 1:(self.ACTION_STICK, "Stick")}
    self.nActions = len(self.actionMapping)
    self.dealerPolicyStickThreshold = 17
    self.deck = range(1,self.N_CARDS_DECK+1)
    self.playerHand = []
    self.dealerHand = []
    state = self.reset()
    
  def step(self, action):
    # Check if the current state is terminal
    reward, done, playerSum, dealerSum, playerHasUsableAce, new_state = self.evaluateGame()
    # Take action for player
    if(action==self.ACTION_HIT):
      # Player hits
      self.playerHand.append(min(self.VAL_FACECARDS, np.random.choice(self.deck)))
      doesPlayerStick = False
    else:
      # Player sticks
      doesPlayerStick = True
    # Select and take action for dealer
    if(dealerSum<self.dealerPolicyStickThreshold):
      # Dealer hits
      self.dealerHand.append(min(self.VAL_FACECARDS, np.random.choice(self.deck)))
      doesDealerStick = False
    else:
      # Dealer sticks
      doesDealerStick = True
    # Calculate and analyse the new state
    if(doesPlayerStick and doesDealerStick):
      # Both stick, hand did not change
      if(dealerSum>playerSum):
        reward = self.reward_lose
        new_state = self.state_lose
      elif(playerSum>dealerSum):
        reward = self.reward_win
        new_state = self.state_win
      else:
        reward = self.reward_draw
        new_state = self.state_draw
      done = True
    else:
      reward, done, playerSum, dealerSum, playerHasUsableAce, new_state = self.evaluateGame()
      
    return new_state, reward, done
    
  def reset(self):
    self.playerHand = []
    self.dealerHand = []
    if(0):
      # Equiprobable random cards
      # Deal cards to player
      self.playerHand.append(min(self.VAL_FACECARDS, np.random.choice(self.deck)))
      self.playerHand.append(min(self.VAL_FACECARDS, np.random.choice(self.deck)))
      # Deal cards to dealer
      self.dealerHand.append(min(self.VAL_FACECARDS, np.random.choice(self.deck)))
      self.dealerHand.append(min(self.VAL_FACECARDS, np.random.choice(self.deck)))
      playerSum, dealerSum, playerHasUsableAce = self.calculateHands()
      state = self.getLinearIndex(playerSum, self.dealerHand[self.IDX_DEALER_CARD_SHOWN], playerHasUsableAce)
    else:
      # Equiprobable random states
      playerSum_state = np.random.choice(self.nStatesPlayerSum)
      playerSum = playerSum_state + self.minPlayerSum
      playerHasUsableAce = np.random.choice(self.nStatesPlayerHasUsableAce)
      if(playerHasUsableAce):
        self.playerHand.append(self.LABEL_ACE)
        self.playerHand.append(playerSum-self.VAL_USABLE_ACE)
      else:
        firstCard = np.random.choice(playerSum-1)+1
        self.playerHand.append(firstCard)
        self.playerHand.append(playerSum-firstCard)
      # Deal cards to dealer
      self.dealerHand.append(min(self.VAL_FACECARDS, np.random.choice(self.deck)))
      self.dealerHand.append(min(self.VAL_FACECARDS, np.random.choice(self.deck)))
      state = self.getLinearIndex(playerSum, self.dealerHand[self.IDX_DEALER_CARD_SHOWN], playerHasUsableAce)
      
    return state
  
  def setHands(self, playerHand, dealerHand): 
    self.playerHand = []
    self.dealerHand = []
    self.playerHand = playerHand
    self.dealerHand = dealerHand
    playerSum, dealerSum, playerHasUsableAce = self.calculateHands()
    state = self.getLinearIndex(playerSum, self.dealerHand[self.IDX_DEALER_CARD_SHOWN], playerHasUsableAce)
    return state
  
  def calculateHands(self):
    if(self.LABEL_ACE in self.playerHand):
      if( (np.sum(self.playerHand) - self.LABEL_ACE + self.VAL_USABLE_ACE)<=self.VAL_BLACKJACK ):
        playerHasUsableAce = self.USABLE_ACE_YES
        playerSum = np.sum(self.playerHand) - self.LABEL_ACE + self.VAL_USABLE_ACE
      else:
        playerHasUsableAce = self.USABLE_ACE_NO
        playerSum = np.sum(self.playerHand) - self.LABEL_ACE + self.VAL_NONUSABLE_ACE
    else:
      playerHasUsableAce = self.USABLE_ACE_NO
      playerSum = np.sum(self.playerHand)
    if(self.LABEL_ACE in self.dealerHand):
      if( (np.sum(self.dealerHand) - self.LABEL_ACE + self.VAL_USABLE_ACE)<=self.VAL_BLACKJACK ):
        dealerSum = np.sum(self.dealerHand) - self.LABEL_ACE + self.VAL_USABLE_ACE
      else:
        dealerSum = np.sum(self.dealerHand) - self.LABEL_ACE + self.VAL_NONUSABLE_ACE
    else:
      dealerSum = np.sum(self.dealerHand)
      
    return playerSum, dealerSum, playerHasUsableAce
    
  def evaluateGame(self):
    playerSum, dealerSum, playerHasUsableAce = self.calculateHands()
    if(playerSum==self.VAL_BLACKJACK and dealerSum==self.VAL_BLACKJACK):
      # Draw
      reward = self.reward_draw
      done = True
      new_state = self.state_draw
    elif(playerSum==self.VAL_BLACKJACK and dealerSum!=self.VAL_BLACKJACK):
      # Player wins
      reward = self.reward_win
      done = True
      new_state = self.state_win
    elif(playerSum!=self.VAL_BLACKJACK and dealerSum==self.VAL_BLACKJACK):
      # Dealer wins
      reward = self.reward_lose
      done = True
      new_state = self.state_lose
    elif(playerSum>self.VAL_BLACKJACK and dealerSum>self.VAL_BLACKJACK):
      # Draw
      reward = self.reward_draw
      done = True
      new_state = self.state_draw
    elif(playerSum<=self.VAL_BLACKJACK and dealerSum>self.VAL_BLACKJACK):
      # Player wins
      reward = self.reward_win
      done = True
      new_state = self.state_win
    elif(playerSum>self.VAL_BLACKJACK and dealerSum<=self.VAL_BLACKJACK):
      # Dealer wins
      reward = self.reward_lose
      done = True
      new_state = self.state_lose
    else:
      reward = self.reward_default
      done = False
      new_state = self.getLinearIndex(playerSum, self.dealerHand[self.IDX_DEALER_CARD_SHOWN], playerHasUsableAce)

    return reward, done, playerSum, dealerSum, playerHasUsableAce, new_state
    
  def getLinearIndex(self, playerSum, dealerShowing, playerHasUsableAce):
    playerSum_state = max(playerSum, self.minPlayerSum)-self.minPlayerSum
    return playerHasUsableAce * self.nStatesPlayerSum * self.nStatesDealerShowing + (dealerShowing-1) * self.nStatesPlayerSum + playerSum_state