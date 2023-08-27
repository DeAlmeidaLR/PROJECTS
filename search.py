# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import searchAgents
import game


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    #The function starts by initializing a stack with a tuple of the start state of the problem and an empty path
    stack = [(problem.getStartState(), [])]
    #then initializes an empty set,  visited, to store states that have already been visited
    visited = set()

    #The main loop of the algorithm pops a node and its path from the top of the stack
    while stack:
        node, path = stack.pop()
        #If the popped node is not a goal state, its state is added to the set of visited states to avoid revisiting it.
        if problem.isGoalState(node):
            return path
        #For each unvisited successor of the popped node, a tuple containing the successor state and an updated path to reach it is pushed onto the stack
        if node not in visited:
            visited.add(node)
            #The algorithm continues until the stack is empty or a goal state is found. If no goal state is found, an empty list is returned.
            for successor in problem.getSuccessors(node):
                stack.append((successor[0], path + [successor[1]]))

    return []

    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    from util import Queue
    #The function starts by initializing a frontier queue with a tuple of the start state of the problem and an empty path
    frontier, visited = Queue(), set()    
    
    #The main loop of the algorithm dequeues a node and its path from the top of the frontier
    frontier.push((problem.getStartState(), []))
    
    while frontier:
        node, path = frontier.pop()
        
        #If the dequeued node is a goal state, the path to reach this goal state is returned.
        if problem.isGoalState(node):
            return path
    #For each unvisited successor of the dequeued node, a tuple containing the successor state and an updated path to reach it is enqueued to the end of the frontier
        if node not in visited:
            visited.add(node)
            for successor in problem.getSuccessors(node):
                #The algorithm continues until the frontier is empty or a goal state is found. If no goal state is found, an empty list is returned.
                frontier.push((successor[0], path + [successor[1]]))

    return []
    
    util.raiseNotDefined()
    

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    from heapq import heappush, heappop
    # Initialize the frontier with a heap
    frontier = []
    visited = set()
    successorsCache = {}

    # Add the start node to the frontier
    startNode = (problem.getStartState(), [], 0)
    heappush(frontier, (0, startNode))

    while frontier:
        # Select the node with the lowest path cost from the frontier
        (priority, node) = heappop(frontier)
        #If the current node is a goal state, it satisfies the condition specified by
        if problem.isGoalState(node[0]):
            #the path to reach this goal state is returned as node[1] . This is because  node[1] contains the path from the start state to the current node
            return node[1]
    #If the current node node is not a goal state, its state is added to the  visited set to avoid revisiting it
        if node[0] not in visited:
            visited.add(node[0])
            successors = successorsCache.get(node[0], None)
            if successors is None:
                successors = list(problem.getSuccessors(node[0]))
                successorsCache[node[0]] = successors
            for successor in successors:
                # Calculate the updated priority for each successor node
                updatedPriority = node[2] + successor[2]
                # The path to reach the successor is also updated by appending the action required to reach it (successor[1]) to the path to reach the current node (node[1])
                updatedPath = node[1] + [successor[1]]
                updatedNode = (successor[0], updatedPath, updatedPriority)
                #This ensures that nodes with lower cost paths are explored first.
                if successor[0] not in visited:
                    heappush(frontier, (updatedPriority, updatedNode))

    # Return an empty list if no solution found
    return []
                
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import PriorityQueue
    #The frontier variable is initialized as a priority queue, where the higher priority corresponds to a lower cost
    #The visited variable is initialized as an empty set.
    frontier, visited = PriorityQueue(), set()
    
    #The algorithm starts by pushing the start state into the frontier with a priority of 0
    #The Node tuple has three fields: state,  empty path, and cost of 0 and push it into the frontier priority queue
    from collections import namedtuple
    Node = namedtuple('Node', ['state', 'path', 'cost'])
    frontier.push(Node(problem.getStartState(), [], 0), 0)
    
    #it enters a loop where it removes the highest priority node from the frontier, updates its path and cost values, and checks whether it is the goal state.
    while not frontier.isEmpty():
        node, path, cost = frontier.pop()
        #If it is the goal state, the algorithm returns the path
        if problem.isGoalState(node):
            return path

        #If the current node is not the goal state, the algorithm adds it to the visited set and expands its successors
        if node not in visited:
            visited.update([node])
            
        #For each successor, the algorithm calculates a new cost and adds it to the frontier if the successor has not been visited before
            for successor, action, step_cost in problem.getSuccessors(node):
                new_cost = cost + step_cost
                if successor not in visited:
                    frontier.push((successor, path + [action], new_cost), new_cost + heuristic(successor, problem))

    return []
                
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
