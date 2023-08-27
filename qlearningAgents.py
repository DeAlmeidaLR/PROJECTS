# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qValues = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qValues.get((state, action), 0.0)
        

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # first obtains all legal actions from the current state using the getLegalActions method and checks whether there are any legal actions available. If there are no legal actions available, the method returns 0.0.
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0
        
        #If there are legal actions, the method initializes a variable maxQ to a very small negative number 
        maxQ = float("-inf")
        #It then loops through all the legal actions and calculates the Q-value for each action using the getQValue method. 
        for action in legalActions:
            qValue = self.getQValue(state, action)
            #If the calculated Q-value for the current action is greater than maxQ, the variable maxQ is updated to the Q-value of the current action.
            if qValue > maxQ:
                maxQ = qValue
        #Finally, the method returns the maxQ value calculated during the loop, which is the maximum Q-value for the given state over all possible actions. 
        return maxQ

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        #Get a list of all legal actions from the current state.
        legalActions = self.getLegalActions(state)
        #Get the maximum Q-value for the given state by calling the computeValueFromQValues method.
        maxQValue = self.computeValueFromQValues(state)
        #Create a new list bestActions containing all actions that have a Q-value equal to maxQValue. This is done using a list comprehension that loops through all legal actions and checks if their Q-value is equal to the maximum Q-value.
        bestActions = [action for action in legalActions if self.getQValue(state, action) == maxQValue]
        #Choose an action from bestActions using the random.choice function. If bestActions is empty return None
        return random.choice(bestActions) if bestActions else None

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        #Retrieves the legal actions available in the current state
        legalActions = self.getLegalActions(state)
        #This is a conditional expression which returns either a randomly chosen legal action or the optimal action.
        #If there are legalActions and util.flipCoin(self.epsilon) is True then random.choice(legalActions) is returned else self.computeActionFromQValues(state) is returned.
        return random.choice(legalActions) if legalActions and util.flipCoin(self.epsilon) else self.computeActionFromQValues(state)
        
        #NB: One could use an if-else statement here, but the combined makes it more brief.

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        qValue = 0
        #Checks if there are any legal actions from the nextState
        if self.getLegalActions(nextState):
          #Loops through each legal action available from the nextState.
          for nextAction in self.getLegalActions(nextState):
            #Updates the qValue by adding the Q-value of the current action to it.
            qValue += self.getQValue(nextState, nextAction)

        #Computes the new Q-value by taking a weighted average of the old Q-value and the learned value
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * qValue)
        
        #NB:For better testing and understanding, one can decrease the learning rate (alpha) to make the agent update its Q-values more slowly, which will make it take longer to converge to an optimal policy

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
    


        """
        Overall, for question 4 and 5, TO UNDERSTAND HOW THE AGENT WORKS, one can:
        1. Decrease the alpha by setting it to a smaller value, such as 0.1. This will cause the agent to update its Q-values more slowly and take longer to converge to an optimal policy
        2. Increase the gamma to make the agent care more about future rewards, which may make it more likely to take suboptimal actions 
        3. Add noise to the rewards to make the agent less certain about the actual reward it receives for each action. This will make it harder for the agent to learn an accurate Q-function and may cause it to explore more.
        """




class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action
    
class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #returns a dictionary of feature values for the given state-action pair.
        features = self.featExtractor.getFeatures(state, action)
        # computes the dot product between the features of a state-action pair and the weights of the linear value function approximation.
        return sum(self.weights[f] * v for f, v in features.items())
  

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # Compute the TD error
        #TD error - which measures the difference between the predicted Q-value and the actual Q-value. It is computed as the sum of the immediate reward and the discounted value of the next state, minus the current estimated Q-value
        qvalue = self.getQValue(state, action)
        next_v = self.getValue(nextState)
        correction = reward + self.discount * next_v - qvalue

        # Update the weights for each feature
        for feature, value in self.featExtractor.getFeatures(state, action).items():
            self.weights[feature] += self.alpha * correction * value
            #We update the weights for each feature because the estimated value of a state is based on the weighted sum of its features
            

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            print(self.weights)
            pass
