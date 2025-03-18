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
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Compute current score
        score = successorGameState.getScore()

        # Distance to the closest food pellet (want this as small as possible)
        if newFood:
            foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood]
            minFoodDistance = min(foodDistances)
        else:
            minFoodDistance = 0  # No food left!

        # Ghost danger - avoid non-scared ghosts that are close!
        ghostPenalty = 0
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            distance = manhattanDistance(newPos, ghostPos)
            if ghostState.scaredTimer == 0:
                if distance <= 1:
                    ghostPenalty += 9999  # Huge penalty if ghost is too close!
            else:
                # Reward for being close to scared ghosts you can eat
                score += 200 / (distance + 1)

        # Combine everything into the final score:
        # - closer food is better (negative because we want smaller distances)
        # - fewer remaining food pellets is better
        # - avoid ghosts unless they’re scared
        finalScore = score
        finalScore += -1.5 * minFoodDistance  # prioritize closer food
        finalScore += -20 * len(newFood)  # fewer food dots left is better
        finalScore -= ghostPenalty  # avoid deadly ghosts

        return finalScore


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

        def minimax(agentIndex, depth, state):
            # Terminal condition: depth limit reached or game over
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            numAgents = state.getNumAgents()

            # Pacman (Maximizing agent)
            if agentIndex == 0:
                bestValue = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = minimax(1, depth, successor)  # Next agent (ghost 1)
                    bestValue = max(bestValue, value)
                return bestValue
            
            # Ghosts (Minimizing agents)
            else:
                bestValue = float('inf')
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    value = minimax(nextAgent, nextDepth, successor)
                    bestValue = min(bestValue, value)
                return bestValue

        # Initial call from Pacman's perspective
        bestScore = float('-inf')
        bestAction = None

        for action in gameState.getLegalActions(0):  # Pacman moves first
            successor = gameState.generateSuccessor(0, action)
            score = minimax(1, 0, successor)  # Start with ghost 1
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num_of_agents = gameState.getNumAgents()-1

        def Maximizing(gameState, curr_depth, alpha, beta):
            if (curr_depth == 0 or gameState.isWin() or gameState.isLose()):  # base case
                return self.evaluationFunction(gameState)
            
            max_eval = -9999999
            legal_actions = gameState.getLegalActions(0)

            for action in legal_actions:
                # play action
                new_board = gameState.generateSuccessor(0, action)

                eval = Minimizing(new_board, curr_depth-1, 1, alpha, beta)

                if (eval > max_eval): max_eval = eval

                if (max_eval > beta): return eval  # alpha-beta cut off
                if (max_eval > alpha): alpha = max_eval
                
            return max_eval

        def Minimizing(gameState, curr_depth, agent, alpha, beta):
            if (gameState.isWin() or gameState.isLose()):  # base case - stop searching
                return self.evaluationFunction(gameState)
            
            min_eval = 9999999

            # get legal actions
            legal_actions = gameState.getLegalActions(agent)

            for action in legal_actions:
                # play action
                new_board = gameState.generateSuccessor(agent, action)

                if (agent == num_of_agents):
                    eval = Maximizing(new_board, curr_depth, alpha, beta)
                else:  # get next ghost
                    eval = Minimizing(new_board, curr_depth, agent+1, alpha, beta)  

                if (eval < min_eval): min_eval = eval

                if (min_eval < alpha): return eval  # alpha-beta cut off
                if (min_eval < beta): beta = min_eval
                
            return min_eval

        search_depth = self.depth-1
        best_action = None

        max_eval = -999999999
        a = -999999

        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:

            eval = Minimizing(gameState.generateSuccessor(0, action), search_depth, 1, a, 999999)
            if (eval > max_eval):
                max_eval = eval
                best_action = action
            
            if (a < eval): a = eval

        return best_action

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
        def expectimax(state, agentIndex, depth):
            # Base case: terminal state or maximum depth reached
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            numAgents = state.getNumAgents()
            
            # Pacman's turn (Maximizer)
            if agentIndex == 0:
                value = float('-inf')
                for action in state.getLegalActions(agentIndex):
                    successor = state.generateSuccessor(agentIndex, action)
                    score = expectimax(successor, 1, depth)  # Pacman's turn moves to ghost
                    value = max(value, score)
                return value
            
            # Ghost's turn (Expectimax)
            else:
                value = 0
                nextAgent = (agentIndex + 1) % numAgents
                nextDepth = depth + 1 if nextAgent == 0 else depth  # Go to next depth after the last ghost

                legalActions = state.getLegalActions(agentIndex)
                numActions = len(legalActions)

                # Calculate the expected value for the ghost's random moves
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    score = expectimax(successor, nextAgent, nextDepth)
                    value += score / numActions  # Average the ghost's moves
                return value
        
        bestScore = float('-inf')
        bestAction = None
        
        # Loop over Pacman's legal actions to find the best one
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            score = expectimax(successor, 1, 0)  # Start with ghost (agentIndex=1)
            if score > bestScore:
                bestScore = score
                bestAction = action
        
        return bestAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    foodList = foodGrid.asList()
    
    # 1. Distance to nearest food
    foodDistances = [manhattanDistance(pacmanPos, food) for food in foodList]
    nearestFoodDistance = min(foodDistances) if foodDistances else 0
    
    # 2. Penalty for being close to non-scared ghosts
    ghostDistances = []
    ghostPenalty = 0
    for ghost, scaredTime in zip(ghostStates, scaredTimes):
        ghostDist = manhattanDistance(pacmanPos, ghost.getPosition())
        ghostDistances.append(ghostDist)
        if scaredTime == 0:
            if ghostDist <= 1:
                ghostPenalty += 100  # Big penalty if Pacman is near a non-scared ghost
            elif ghostDist <= 3:
                ghostPenalty += 10  # Smaller penalty if Pacman is not too close

    # 3. Favor being close to scared ghosts
    scaredGhostsBonus = 0
    for ghost, scaredTime in zip(ghostStates, scaredTimes):
        if scaredTime > 0:
            scaredGhostsBonus += 5 / (1 + min(ghostDistances))  # Favor getting closer to scared ghosts
    
    # 4. Number of remaining food pellets
    remainingFood = len(foodList)
    
    # Calculate final score
    score = currentGameState.getScore()
    score -= nearestFoodDistance * 2  # Pacman prefers getting close to food
    score -= ghostPenalty  # Penalize proximity to ghosts
    score += scaredGhostsBonus  # Favor getting close to scared ghosts
    score -= remainingFood * 2  # Prefer less food remaining
    
    return score

# Abbreviation
better = betterEvaluationFunction

