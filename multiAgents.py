# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Calculate the distances to the nearest food and ghost using the manhattanDistance function, which calculates the Manhattan distance between two points.
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]

        # If there's food nearby, prioritize eating it
        minFoodDistance = min(foodDistances or [float('inf')])
        # If there is nearby food, the foodScore is calculated as 1.0 / max(minFoodDistance, 1) where minFoodDistance is the distance to the nearest food. This means that the closer the food is to Pac-Man, the higher the foodScore.
        foodScore = 1.0 / max(minFoodDistance, 1)

        # If there's a ghost nearby, prioritize avoiding it
        minGhostDistance = min(ghostDistances or [float('inf')])
        # If there is a nearby ghost, the ghostScore is calculated as -1.0 / max(minGhostDistance, 1). This means that the closer the ghost is to Pac-Man, the lower the ghostScore.
        # In practice, Pac-Man is trying to maximize its score by avoiding a lower score.
        ghostScore = -1.0 / max(minGhostDistance, 1)
        #NB: If we used a different negative value like -2, the effect of the ghost distance on the score would be less severe, so Pacman might not avoid the ghost as strongly.

        # Calculate the total score
        score = successorGameState.getScore() + foodScore + ghostScore

        # SuccessorGameState.getScore() is the score of the game state after Pac-Man takes an action. It includes points for eating food, and killing ghosts. foodScore encourages Pac-Man to eat the remaining food pellets on the board.
        return score
        return successorGameState.getScore()
    
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        def minimax(gameState, depth, agentIndex):
            # If the search depth is zero, or if the game is already won or lost, the minimax function returns the score of the current game state as calculated by the evaluationFunction
            if depth == 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # The bestScore variable stores the best score found so far.
            # When the algorithm is searching for the maximum score for the agent, any score encountered in the loop will be greater than float('-inf'), so it will be updated as the new bestScore
            # Then initializes the variable bestScore to negative infinity (-inf) if agentIndex is equal to 0, otherwise it initializes it to positive infinity - (minimax)
            bestScore = float('-inf') if agentIndex == 0 else float('inf')
            #NB: when initializing the best score, we set it to negative infinity for the maximizing agent and positive infinity for the minimizing agent to ensure that the first evaluation of a game state will always improve the score for the maximizing agent and decrease the score for the minimizing agent.

            # for action in gameState.getLegalActions(agentIndex) loops through all the legal actions that the current player (represented by agentIndex) can take in the current gameState.
            for action in gameState.getLegalActions(agentIndex):
                nextGameState = gameState.generateSuccessor(agentIndex, action)

                # nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents() calculates the index of the next player in the game by taking the remainder of (agentIndex + 1) divided by the total number of agents in the game
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

                # This recursively calls the minimax function on the nextGameState, with the depth decreased by 1 if the next player is the first player because the depth parameter represents the number of moves that are left to be searched. In the minimax algorithm, each recursive call to the minimax function represents a single move made by one of the players.
                score = minimax(nextGameState, depth - (nextAgentIndex == 0), nextAgentIndex)
                # This updates the bestScore based on whether the current player is the maximizing player or the minimizing player. If the current player is the maximizing player, then bestScore is updated to the maximum of bestScore and score. Otherwise, bestScore is updated to the minimum of bestScore and score.
                bestScore = max(bestScore, score) if agentIndex == 0 else min(bestScore, score)
            #returns the best score
            return bestScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(state, agent_index, depth, alpha, beta):
        # Function first checks if the current state is a terminal state (win, lose, or depth limit reached), and if so, returns the evaluation function value and None for the best action
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None
            # Used to initialize a variable that will later hold the maximum or minimum value of a set of numbers.
            value = float('-inf')

            # Best_action is used to keep track of the action that leads to the maximum or minimum value found so far during the recursive search. In the max_value function, best_action is initially set to None, and is updated to the action that leads to the maximum value found so far during the search
            best_action = None

            # This is a loop that iterates over all the legal actions that the current agent can take in the current state
            for action in state.getLegalActions(agent_index):
                # For each action, a successor state is generated by applying the action to the current state. The resulting state is stored in successor_state
                successor_state = state.generateSuccessor(agent_index, action)
                # The min_value function is called recursively with the successor_state as the new state, the next agent as the agent_index + 1, and the current depth, alpha, and beta values. The new_value returned by this recursive call represents the value of the new state.
                # Is a function that returns a tuple of two values, therefore, new_value will contain the utility value of the node.
                new_value, _ = min_value(successor_state, agent_index + 1, depth, alpha, beta)
                # If new_value is greater than value, then the value variable, which stores the maximum value found so far, is updated to new_value. Additionally, the best_action variable, which stores the action that leads to the maximum value found so far, is updated to action
                if new_value > value:
                    value = new_value
                    best_action = action
                # If v is greater than beta, then the maximum value found so far is greater than the maximum value that the parent node can achieve, so the search can be terminated early. The current maximum value (v) and the corresponding action (best_action) are returned.
                if value > beta:
                    return value, best_action
                # Alpha is the best value found so far by the maximizing player on the path from the root of the game tree to the current node. It represents the lower bound on the possible values of the current node.
                # The alpha value is updated to be the maximum value found so far.
                alpha = max(alpha, value)
            # Return value, best_action - The maximum value found so far and the corresponding action are returned
            return value, best_action
        
        # It computes the minimum value that the minimizing player can achieve at a given game state, assuming that both players play optimally.
        # The function takes as input the current game state, the index of the current agent, the depth of the search tree, and the alpha and beta values that define the bounds of the search
        def min_value(state, agent_index, depth, alpha, beta):
            # If the game is in a terminal state or the search has reached the maximum depth, the evaluation function is called to compute the score of the current state, and this score is returned along with None as the best action (since there is no action to take at a terminal state).
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None
            # Otherwise, the function initializes the value variable to infinity, and iterates over all the legal actions that the current agent can take at the current state
            value = float('inf')
            best_action = None
            # Is a line of code that calculates the index of the next agent that will take an action in the game.
            next_agent = (agent_index + 1) % state.getNumAgents()

            # This is a loop that iterates over all the legal actions that the current agent can take in the current state
            for action in state.getLegalActions(agent_index):
                successor_state = state.generateSuccessor(agent_index, action)
                # Checks if the next agent is the first one in the game. If it is, the next level in the tree search is a maximizing level, and the function calls max_value to determine the value of the successor state.
                if next_agent == 0:
                    new_value, _ = max_value(successor_state, next_agent, depth + 1, alpha, beta)
                # Otherwise, the next level in the tree search is a minimizing level, and the function calls min_value to determine the value of the successor state.
                else:
                    new_value, _ = min_value(successor_state, next_agent, depth, alpha, beta)
                #If new_value is smaller than the current best value value, value is updated to new_value and best_action is updated to the current action
                if new_value < value:
                    value = new_value
                    best_action = action
                # Checks if the value of the current state is less than the minimum value found so far in a maximizing level, indicated by alpha. If it is, it means that the parent maximizing level would never choose this state because it can do at least as well with a different action.
                if value < alpha:
                    return value, best_action
                # beta is the minimum score that the maximizing player can achieve, so if a minimizing player has found a state with a score less than or equal to beta, the maximizing player would not choose that action because it can achieve at least beta with another action.
                beta = min(beta, value)
            # After iterating through all the legal actions, the function returns the best value and action found
            return value, best_action
        
        #In Python, the underscore character (_) is often used as a variable name when a value is returned from a function or method, but the specific value is not actually used in the code
        _, action = max_value(gameState, self.index, 0, float('-inf'), float('inf'))
        return action
      
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        """The get_max_value function calculates the maximum value for the current player, given the current game state and the current search depth.
        If the game is over (i.e., the current player has won or lost) or the maximum search depth has been reached, the function returns the evaluation of the current state and None.
        Otherwise, the function loops over all legal actions for the current player and recursively calls the get_exp_value function to get the expected value for the opponent's next move.
        The best action and its corresponding score are updated based on the maximum score among all possible actions. Finally, the function returns the best score and best action."""
        #Takes the current game state, the current agent's index, and the current depth of the search tree as input.
        def get_max_value(state, agent_index, depth):
            #The function checks if the game state is a win or loss state or if the depth limit has been reached. If so, the function returns the score of the state and None
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None
            
            #If the game state is not a win or loss state and the depth limit has not been reached, the function generates all legal successor states of the current state for the current agent
            scores_actions = [(get_expected_value(state.generateSuccessor(agent_index, action), agent_index + 1, depth)[0], action) for action in state.getLegalActions(agent_index)]
            max_score, best_action = max(scores_actions, key=lambda x: x[0])
            return max_score, best_action
        
        #takes the current game state, the current agent's index, and the current depth of the search tree as input.
        def get_expected_value(state, agent_index, depth):
            #The function checks if the game state is a win or loss state or if the depth limit has been reached. If so, the function returns the score of the state and None.
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None
            #If the game state is not a win or loss state and the depth limit has not been reached, the function generates all legal successor states of the current state for the current agent.
            num_actions = len(state.getLegalActions(agent_index))
            i = 0
            average_score = 0

            while i < num_actions:
                #The function calls get_max_value() function for each of these successor states to get the maximum score of each state
                #then calculates the average score of all the maximum scores obtained from the successor states and returns this average score along with None.
                action = state.getLegalActions(agent_index)[i]
                #checks whether the current agent is the last agent in the game. If this is the case, it means that it is time to evaluate the state (i.e., get its max value). 
                if agent_index == state.getNumAgents() - 1:
                    #gets the max value of the state that results from taking the current action in the current state.
                    score, _ = get_max_value(state.generateSuccessor(agent_index, action), 0, depth + 1)
                else:
                        score, _ = get_expected_value(state.generateSuccessor(agent_index, action), agent_index + 1, depth)
                average_score += score / num_actions
                
                #updates the index of the current action by incrementing it by one so that the next legal action can be evaluated
                i += 1

            return average_score, None
        #Finally, the best_score and best_action are obtained by calling get_max_value() function on the initial game state, the current player's index, and the current depth of the search tree.
        best_score, best_action = get_max_value(gameState, self.index, 0)
        #The function returns the best action to take based on the calculated best score
        return best_action
      
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      1. Extract the current position of Pac-Man, the layout of the maze (including the position of food pellets), and the position and state of the ghosts from the input currentGameState object.
      2. Compute the Manhattan distance between Pac-Man and each food pellet in the maze. 
      3. Determine the closest food pellet to Pac-Man by finding the minimum value in the foodDistances list.
      4. Compute the number of food pellets left in the game by calling the getNumFood() method of the currentGameState object
      5. Compute the Manhattan distance between Pac-Man and each ghost in the maze, as well as the amount of time each ghost will remain scared.
      6. Compute the score of the current game state by using a weighted evaluation function.
    """
    "*** YOUR CODE HERE ***"
    
    # The function takes a currentGameState as input, which contains information about the current state of the game, including the position of Pac-Man, the position of the food pellets, and the position of the ghosts.
    pacmanPosition = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    # Calculates the Manhattan distance between the current position of Pac-Man and each food pellet present in the foodGrid. The foodGrid is an instance of Grid, which represents the layout of the maze and the positions of the food pellets.
    # It then calculates the distance to the closest food pellet by computing the Manhattan distance (a distance metric that only considers horizontal and vertical distances) between Pac-Man's current position and each food pellet in the grid, and taking the minimum distance.
    # After this loop, foodDistances contains the Manhattan distances between Pac-Man and all the food pellets present in the maze. 
    foodDistances = [manhattanDistance(pacmanPosition, food) for food in foodGrid.asList()] 
    closestFood = min(foodDistances) if foodDistances else 0

    # Calculate the number of food pellets left
    numFoodLeft = currentGameState.getNumFood()

    # Calculate the distance to the closest ghost and its scared timer
    ghostDistances = []
    for ghostState in ghostStates:

        # It then calculates the distance to the closest ghost and its scared timer by iterating over the list of ghostStates, computing the Manhattan distance between Pac-Man's current position and each ghost's position, and storing the closest distance and the scared timer of that ghost.
        distance = manhattanDistance(pacmanPosition, ghostState.getPosition())

        # In the context of a Pac-Man game, the scaredTimer variable represents the number of moves that the ghosts will remain in a "scared" state, during which they can be eaten by Pac-Man for points.
        scaredTimer = ghostState.scaredTimer

        # The is adding a tuple containing two values to the ghostDistances list
        # By appending a tuple with both the distance and the scared timer to the ghostDistances list, the code is keeping track of both pieces of information for each ghost.
        ghostDistances.append((distance, scaredTimer))

    # The min function returns the tuple with the smallest distance value, which contains both the distance to the closest ghost and the scared timer for that ghost
    # This allows the evaluation function to consider the distance to the closest ghost as well as the amount of time the ghost will remain scared when calculating the score.
    closestGhost, scaredTimer = min(ghostDistances, default=(0, 0))

    # Weighted evaluation function
    """" Overall, this evaluation function tries to maximize the score by encouraging Pac-Man to eat food pellets, avoid ghosts, and avoid spending too much time around scared ghosts.
    1 / (closestFood + 1): this term increases the score if Pac-Man is closer to a food pellet. The closestFood variable is added by 1 to prevent division by zero if Pac-Man is adjacent to a food pellet.
    1 / (numFoodLeft + 1): this term increases the score if there are fewer food pellets remaining in the game. The numFoodLeft variable is added by 1 to prevent division by zero if there are no food pellets left.
    - 5 / (closestGhost + 1): this term decreases the score if Pac-Man is closer to a ghost. The closestGhost variable is added by 1 to prevent division by zero if Pac-Man is adjacent to a ghost. The negative sign indicates that being closer to a ghost is bad for Pac-Man.
    - 5 * scaredTimer: this term decreases the score if the scared timer for the closest ghost is greater than 0. The more scared time remaining, the lower the score. """
    score = currentGameState.getScore() \
        + 1 / (closestFood + 1) \
        + 1 / (numFoodLeft + 1) \
        - 5 / (closestGhost + 1) \
        - 5 * scaredTimer
    

    # The score is calculated as the sum of different factors that are important to evaluate the state of the game
    # returns the calculated score as the evaluation of the current game state.
    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

