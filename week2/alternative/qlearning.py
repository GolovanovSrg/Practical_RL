# qlearningAgents.py
# ------------------
## based on http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random, math
import numpy as np
from collections import defaultdict

class QLearningAgent():
  """
    Q-Learning Agent

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate aka gamma)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions for a state
      - self.getQValue(state,action)
        which returns Q(state,action)
      - self.setQValue(state,action,value)
        which sets Q(state,action) := value
    
    !!!Important!!!
    NOTE: please avoid using self._qValues directly to make code cleaner
  """

  def __init__(self,alpha,epsilon,discount,getLegalActions):
    "We initialize agent and Q-values here."
    
    self.getLegalActions= getLegalActions
    self._qValues = defaultdict(lambda:defaultdict(lambda:0))
    self.alpha = alpha
    self.epsilon = epsilon
    self.discount = discount

  def getQValue(self, state, action):
    """
      Returns Q(state,action)
    """
    
    return self._qValues[state][action]

  def setQValue(self,state,action,value):
    """
      Sets the Qvalue for [state,action] to the given value
    """
    
    self._qValues[state][action] = value

  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.
    """
    
    possibleActions = self.getLegalActions(state)
    if len(possibleActions) == 0:
        return 0.0
    return np.amax([self.getQValue(state, a) for a in possibleActions])
  
  def getPolicy(self, state):
    """
      Compute the best action to take in a state. 
    """
    
    possibleActions = self.getLegalActions(state)
    if len(possibleActions) == 0:
        return None
    best_action_idx = np.argmax([self.getQValue(state, a) for a in possibleActions])
    return possibleActions[best_action_idx]

  def getAction(self, state):
    """
      Compute the action to take in the current state, including exploration.  
      
      With probability self.epsilon, we should take a random action.
      otherwise - the best policy action (self.getPolicy).

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    
    possibleActions = self.getLegalActions(state)
    if len(possibleActions) == 0:
        return None

    if random.random() < self.epsilon:
        return random.choice(possibleActions)
    
    return self.getPolicy(state)

  def update(self, state, action, nextState, reward):
    """
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    
    gamma = self.discount
    learning_rate = self.alpha  
    reference_qvalue = reward + self.getValue(nextState)
    updated_qvalue = (1 - learning_rate) * self.getQValue(state, action) + learning_rate * reference_qvalue
    self.setQValue(state, action, updated_qvalue)
