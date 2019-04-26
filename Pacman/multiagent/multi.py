# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from time import sleep

from util import manhattanDistance, Queue
from game import Directions
import random, util

from game import Agent

INF = 999999


def bfsDist(m, start):
    """Find bfs distance from 'start' to any point in map 'm'."""
    q = Queue()
    q.push(start)
    m[start[0]][start[1]] = 0

    dvs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    while not q.isEmpty():
        current = q.pop()
        dist = m[current[0]][current[1]]

        for dv in dvs:
            pos = (current[0] + dv[0], current[1] + dv[1])
            if pos[0] >= 0 and pos[0] < m.width and \
                    pos[1] >= 0 and pos[1] < m.height and \
                    type(m[pos[0]][pos[1]]).__name__ == 'bool' and \
                    m[pos[0]][pos[1]] == False:
                #: write distance directly to map m
                m[pos[0]][pos[1]] = dist + 1
                q.push(pos)
    return m


bfsDistCache = {}


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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (oldFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        global bfsDistCache

        gameState = currentGameState.generateSuccessor(0, action)

        currentPos = gameState.getPacmanPosition()
        oldfood = gameState.getFood().asList() + gameState.getCapsules()

        try:  #: creation of bfs map
            bfsDists = bfsDistCache[currentPos]
        except:
            bfsDists = bfsDistCache[currentPos] = bfsDist(gameState.getWalls().deepCopy(), currentPos)

        #: evaluation baseline:
        ret = gameState.getScore()

        #: list of (timer, position) of all scared ghosts
        ghostScaredTimer = filter(lambda i: i[0] > 0,
                                  [(s.scaredTimer, s.getPosition()) for s in gameState.getGhostStates()])

        #: delete scared ghosts that are not reachible (cannot get to ghost in-time)
        ghostsFoodDist = map(lambda i: i[1],
                             filter(lambda d: d[1] < d[0],  # d = (scaredTimer, bfsDist)
                                    [(s[0], bfsDists[int(s[1][0])][int(s[1][1])]) for s in ghostScaredTimer]))

        #: scared ghosts modifies score
        if len(ghostsFoodDist) > 0:
            ret += 10.0 / min(ghostsFoodDist)

        #: normal food
        fooddist = [bfsDists[food[0]][food[1]] for food in oldfood]
        if len(fooddist) > 0:
            ret += 1.0 / min(fooddist)

        return ret


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def miniMax(self, gameState, depth, agentIndex=0):
        """
          Return the best choice (score, action) for the current agent.
          If agentIndex == 0 it is a max node, otherwise it is a min node.
        """

        #: check if the game ends, or we reach the depth
        if gameState.isWin() or gameState.isLose() or depth == 0:
            #: return (current_score, )
            return (self.evaluationFunction(gameState),)

        numAgents = gameState.getNumAgents()
        #: if current agent is the last agent in game, decrease the depth
        newDepth = depth if agentIndex != numAgents - 1 else depth - 1
        newAgentIndex = (agentIndex + 1) % numAgents

        #: actionlist = [(expectations, action) for each legal actions]
        actionList = [ \
            (self.miniMax(gameState.generateSuccessor(agentIndex, a), \
                          newDepth, newAgentIndex)[0], a) for a in gameState.getLegalActions(agentIndex)]

        if (agentIndex == 0):  #: max node
            return max(actionList)  #: return action that gives max score
        else:  #: min node
            return min(actionList)  #: return action that gives min score

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        return self.miniMax(gameState, self.depth)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabetaUtil(self, depth, agent):
        pass

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        PACMAN = 0

        def max_agent(state, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return state.getScore()
            actions = state.getLegalActions(PACMAN)
            best_score = float("-inf")
            best_action = Directions.STOP
            for action in actions:
                score = min_agent(state.generateSuccessor(PACMAN, action), depth, 1, alpha, beta)
                if score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if best_score > beta:
                    return best_score
            if depth == 0:
                return best_action
            else:
                return best_score

        def min_agent(state, depth, ghost, alpha, beta):
            if state.isLose() or state.isWin():
                return state.getScore()
            next_ghost = ghost + 1
            if ghost == state.getNumAgents() - 1:
                # Although I call this variable next_ghost, at this point we are referring to a pacman agent.
                # I never changed the variable name and now I feel bad. That's why I am writing this guilty comment :(
                next_ghost = PACMAN
            actions = state.getLegalActions(ghost)
            best_score = float("inf")
            score = best_score
            for action in actions:
                if next_ghost == PACMAN:  # We are on the last ghost and it will be Pacman's turn next.
                    if depth == self.depth - 1:
                        score = self.evaluationFunction(state.generateSuccessor(ghost, action))
                    else:
                        score = max_agent(state.generateSuccessor(ghost, action), depth + 1, alpha, beta)
                else:
                    score = min_agent(state.generateSuccessor(ghost, action), depth, next_ghost, alpha, beta)
                if score < best_score:
                    best_score = score
                beta = min(beta, best_score)
                if best_score < alpha:
                    return best_score
            return best_score

        return max_agent(gameState, 0, float("-inf"), float("inf"))


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphaBeta(self, gameState, depth, agentIndex=0, alpha=(-INF,), beta=(INF,)):
        """
          Return the best choice (score, action) for the current agent.
          If agentIndex == 0 it is a max node, otherwise it is a min node.
        """
        #: check if the game ends, or we reach the depth
        if gameState.isWin() or gameState.isLose() or depth == 0:
            #: return (current_score, )
            print gameState
            return self.evaluationFunction(gameState),None

        num_agents = gameState.getNumAgents()
        #: if current agent is the last agent in game, decrease the depth
        new_depth = depth if agentIndex != num_agents - 1 else depth - 1
        new_agent_index = (agentIndex + 1) % num_agents

        if (agentIndex == 0):
            v = (-INF,)
            for action in gameState.getLegalActions(agentIndex):
                next_state = gameState.generateSuccessor(agentIndex, action)
                v = max([v, (self.alphaBeta(next_state, new_depth, new_agent_index, alpha, beta)[0], action)])
                if v >= beta:
                    return v
                alpha = max([alpha, v])
            return alpha

        else:  #: min node
            v = (INF,)
            for action in gameState.getLegalActions(agentIndex):
                next_state = gameState.generateSuccessor(agentIndex, action)
                v = min([v, (self.alphaBeta(next_state, new_depth, new_agent_index, alpha, beta)[0], action)])
                if alpha >= v:
                    return v
                beta = min([beta, v])
            return beta

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        s = self.alphaBeta(gameState, self.depth)
        print "SCORE", s[0]
        return s[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, gameState, depth, agentIndex=0):
        #: check if the game ends, or we reach the depth
        if gameState.isWin() or gameState.isLose() or depth == 0:
            #: return (current_score, )
            return (self.evaluationFunction(gameState),)

        numAgents = gameState.getNumAgents()
        #: if current agent is the last agent in game, decrease the depth
        newDepth = depth if agentIndex != numAgents - 1 else depth - 1
        newAgentIndex = (agentIndex + 1) % numAgents
        legalActions = gameState.getLegalActions(agentIndex)

        try:
            legalActions.remove(DIRECTIONS.STOP)
        except:
            pass

        #: actionlist = [(expectations, action) for each legal actions]
        actionList = [ \
            (self.expectimax(gameState.generateSuccessor(agentIndex, a), \
                             newDepth, newAgentIndex)[0], a) for a in gameState.getLegalActions(agentIndex)]

        if (agentIndex == 0):  #: max node
            #: return (max_expectation, best_action)
            return max(actionList)
        else:  #: chance node
            #: return (new_expectation, )
            #: for which 'new_expectation' means average expectation of leagal actions
            return (reduce(lambda s, a: s + a[0], actionList, 0) / len(legalActions),)

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        return self.expectimax(gameState, self.depth)[1]


from util import Queue


def bfsDist(m, start):
    """Find bfs distance from 'start' to any point in map 'm'."""
    q = Queue()
    q.push(start)
    m[start[0]][start[1]] = 0

    dvs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    while not q.isEmpty():
        current = q.pop()
        dist = m[current[0]][current[1]]

        for dv in dvs:
            pos = (current[0] + dv[0], current[1] + dv[1])
            if pos[0] >= 0 and pos[0] < m.width and \
                    pos[1] >= 0 and pos[1] < m.height and \
                    type(m[pos[0]][pos[1]]).__name__ == 'bool' and \
                    m[pos[0]][pos[1]] == False:
                #: write distance directly to map m
                m[pos[0]][pos[1]] = dist + 1
                q.push(pos)
    return m


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:

        The score returned by this evaluation function is defined as follows:

        score = <current state score> + 1/<dist to nearest food> + 10/<dist to nearest scared ghost>

        if there are no scared ghosts, the last term is 0.

        The term <current state score> should dominant the following terms because it it this score
        that the pacman want to maximize. By putting the distance values in the denominator, the
        maximum impact on the final score for the distance valus is N, where N is the nominator of
        the term. The nominator value of 1 and 10 are chosen imperically, which tend to make ghost-
        hunting more vital.

    """
    global bfsDistCache

    currentPos = currentGameState.getPacmanPosition()
    oldfood = currentGameState.getFood().asList() + currentGameState.getCapsules()

    try:  #: creation of bfs map
        bfsDists = bfsDistCache[currentPos]
    except:
        bfsDists = bfsDistCache[currentPos] = bfsDist(currentGameState.getWalls().deepCopy(), currentPos)

    #: evaluation baseline:
    ret = currentGameState.getScore()

    #: list of (timer, position) of all scared ghosts
    ghostScaredTimer = filter(lambda i: i[0] > 0,
                              [(s.scaredTimer, s.getPosition()) for s in currentGameState.getGhostStates()])

    #: delete scared ghosts that are not reachible (cannot get to ghost in-time)
    ghostsFoodDist = map(lambda i: i[1],
                         filter(lambda d: d[1] < d[0],  # d = (scaredTimer, bfsDist)
                                [(s[0], bfsDists[int(s[1][0])][int(s[1][1])]) for s in ghostScaredTimer]))

    #: scared ghosts modifies score
    if len(ghostsFoodDist) > 0:
        ret += 10.0 / min(ghostsFoodDist)

    #: normal food
    fooddist = [bfsDists[food[0]][food[1]] for food in oldfood]
    if len(fooddist) > 0:
        ret += 1.0 / min(fooddist)

    return ret


# Abbreviation
better = betterEvaluationFunction
