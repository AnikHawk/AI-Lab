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




#   python pacman.py -l bigMaze -z 0.5 -p SearchAgent -a fn=ucs --frameTime 0
#   python pacman.py -l mediumMaze -z 0.8 -p SearchAgent -a fn=ucs --frameTime 0
def depthFirstSearch(problem):
    stack = util.Stack()
    iniState = problem.getStartState()
    visited = []
    actions = []
    stack.push((iniState, actions))

    while not stack.isEmpty():
        state, actions = stack.pop()
        visited.append(state)

        if(problem.isGoalState(state)): return actions

        children = problem.getSuccessors(state)
        for child in children:
            nextState = child[0]
            nextDirection = child[1]
            if nextState not in visited:
                stack.push((nextState, actions+[nextDirection]))



def breadthFirstSearch(problem):
    que = util.Queue()
    iniState = problem.getStartState()
    visited = []
    actions = []
    que.push((iniState, actions))

    while not que.isEmpty():
        state, actions = que.pop()
        visited.append(state)

        if problem.isGoalState(state): return actions

        children = problem.getSuccessors(state)
        for child in children:
            nextState = child[0]
            nextDirection = child[1]
            if nextState not in visited:
                que.push((nextState, actions + [nextDirection]))
                visited.append(nextState)




def uniformCostSearch(problem):
    pq = util.PriorityQueue()
    iniState = problem.getStartState()
    actions = []
    cost = 0
    visited = []
    pq.push((iniState,actions),cost)
    successorDictionary = {}

    while not pq.isEmpty():
        state, actions = pq.pop()
        visited.append(state)

        if problem.isGoalState(state): return actions

        if state in successorDictionary:
            children = successorDictionary[state]
        else:
            children = problem.getSuccessors(state)
            successorDictionary[state] = children

        for child in children:
            nextState = child[0]
            nextDirection = child[1]
            if nextState not in visited:
                new_actions = actions + [nextDirection]
                pq.push((nextState,new_actions),problem.getCostOfActions(new_actions))



def iterativeDeepeningSearch(problem):
    stack = util.Stack()
    limit = 1

    while True:
        iniState = problem.getStartState()
        actions = []
        visited = []
        depth = 0
        stack.push((iniState,actions,depth))
        (state,actions,depth) = stack.pop()
        visited.append(state)

        while not problem.isGoalState(state):
            children = problem.getSuccessors(state)
            for child in children:
                nextState = child[0]
                nextDirection = child[1]
                #nextDepth = child[2]
                if (not nextState in visited) and (depth + 1 <= limit):
                    stack.push((nextState, actions + [nextDirection], depth + 1))
                    visited.append(nextState)

            if stack.isEmpty():break
            (state, actions, depth) = stack.pop()

        if problem.isGoalState(state):
            return actions

        limit += 1




#   python pacman.py -l bigMaze -z 0.5 -p SearchAgent -a fn=ucs --frameTime 0
#   python pacman.py -l mediumMaze -z 0.8 -p SearchAgent -a fn=ucs --frameTime 0

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    pq = util.PriorityQueue()
    iniState = problem.getStartState()
    visited = []
    actions = []
    h_cost = heuristic(iniState, problem)
    g_cost = 0

    pq.push((iniState,actions,g_cost),g_cost+h_cost)
    (state,actions,g_cost) = pq.pop()
    visited.append((iniState, g_cost))

    while not problem.isGoalState(state):
        successors = problem.getSuccessors(state)
        for child in successors:
            nextState = child[0]
            nextDirection = child[1]
            nextCost = child[2]
            isNextVisited = False
            h_cost = heuristic(nextState,problem)

            total_cost = g_cost + nextCost
            for (visitedState, visitedCost) in visited:
                if (nextState == visitedState) and (total_cost >= visitedCost):
                    isNextVisited = True

            if not isNextVisited:
                pq.push((nextState, actions + [nextDirection], g_cost+nextCost),
                            g_cost + nextCost + h_cost)
                visited.append((nextState, g_cost + nextCost))

        (state, actions, g_cost) = pq.pop()
        if problem.isGoalState(state):
            return actions


def bestFirstSearch(problem,heuristic=nullHeuristic):
    pq = util.PriorityQueue()
    iniState = problem.getStartState()
    actions = []
    h_cost = heuristic(iniState,problem)
    visited = []
    pq.push((iniState, actions), h_cost)

    while not pq.isEmpty():
        state, actions = pq.pop()
        visited.append(state)

        if problem.isGoalState(state): return actions

        children = problem.getSuccessors(state)
        for child in children:
            nextState = child[0]
            nextDirection = child[1]
            h_cost = heuristic(nextState,problem)
            if nextState not in visited:
                new_actions = actions + [nextDirection]
                pq.push((nextState, new_actions), h_cost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
bestfs = bestFirstSearch


