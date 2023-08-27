# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        #This initializes the iteration counter and a util.Counter() object to store the new value estimates.
        new_values = util.Counter()
        #The for loop executes for self.iterations number of times
        for _ in range(self.iterations):
            #In the for loop, the code iterates over each state in the MDP using the getStates() method.
            for state in self.mdp.getStates():
                #The code checks if the current state is a terminal state using the isTerminal() method. If so, it sets the new value for the state to 0.
                """
                A terminal state is where the agent cannot take any further actions and the code ends.
                When the algorithm encounters a terminal state, it sets the value estimate for that state to 0 because there are no future rewards from that state.
                By checking if a state is terminal, the algorithm can accurately update the value estimates for all states in the MDP, including those that cannot lead to any further rewards
                """
                new_values[state] = 0 if self.mdp.isTerminal(state) else max(sum(prob * (self.mdp.getReward(state, action, next_state) + self.discount * self.values[next_state]) for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action)) for action in self.mdp.getPossibleActions(state))
    
            self.values, new_values = new_values, util.Counter()

            new_values = util.Counter()
            #This process is repeated until the number of iterations specified by self.iterations has been reached. At the end of the algorithm, self.values contains the optimal value estimates for each state in the MDP.


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #This initializes the Q-value estimate to zero.
        #Initially, we have no information about the Q-values of any state-action pair. In Q-learning, the agent learns by updating the Q-values based on the rewards it receives from the environment.
        #Before any learning takes place, we assume that the expected value of the Q-value for any state-action pair is 0
        q_value = 0
        #Retrieves the list of possible next states and transition probabilities for the given state-action pair, and initializes some variables for iterating over them.
        tsp_list = self.mdp.getTransitionStatesAndProbs(state, action)

        #Loops over each possible next state and transition probability, and computes the expected value of the Q-value for that state by multiplying the transition probability by the (reward + discounted next state value), and adding it to the running sum of Q-values. The resulting sum is the estimated Q-value for the given (state, action) pair
        for next_state, prob in tsp_list:
            reward = self.mdp.getReward(state, action, next_state)
            q_value += prob * (reward + self.discount * self.values[next_state])

        #This returns the estimated Q-value for the given (state, action) pair.
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #This retrieves the list of possible actions for the given state, and checks if there are any legal actions. If there are no legal actions (e.g., if the state is terminal), the method returns None.
        #We retrieve the list of possible actions for the given state in order to select the action with the highest Q-value
        legal_actions = self.mdp.getPossibleActions(state)
        if not legal_actions:
            return None
        #We compute the Q-value for each possible action using the computeQValueFromValues method, which estimates the Q-value for a given (state, action) pair. By selecting the action with the highest Q-value, we are choosing the action that is most likely to lead to a high cumulative reward in the future
        return max(legal_actions, key=lambda a: self.computeQValueFromValues(state, a))


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
