import random

import numpy
import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """
        # todo 1

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        # board = successor_game_state.board
        # max_tile = successor_game_state.max_tile
        # score = successor_game_state.score
        # todo - maybe improve this as well if scores being given

        "*** YOUR CODE HERE ***"
        s = successor_game_state

        cor = max_in_sw_corner(s)
        occ = num_occupied(s)
        sco = score(s)
        opn = 1 - (occ / 16)
        sum_mono = sum_monotones(s)

        riv =  (s.score / occ) * s.max_tile

        mono3 = sco * (4*cor + 1*sum_mono + 1*opn)  # 20g/depth1: med=? avg=?   10g/depth2: med=? avg=?

        return mono3


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        """*** YOUR CODE HERE ***"""
        # todo 2
        action, val = self.minimax(game_state, 0, self.depth)
        # print(val)
        # exit(0)
        return action




    def minimax(self, game_state, agent_index, depth):
        legal_actions = game_state.get_legal_actions(agent_index)
        if depth == 0 or len(legal_actions) == 0:
            return Action.STOP, self.evaluation_function(game_state)
        actions = {}
        for action in legal_actions:
            state = game_state.generate_successor(agent_index, action)
            new_action, value = self.minimax(state, 1 - agent_index, depth - 0.5)
            actions[action] = value
        if agent_index == 0:
            best_action = max(actions, key=lambda k: actions[k])
            return best_action, actions[best_action]
        else:
            best_action = min(actions, key=lambda k: actions[k])
            return best_action, actions[best_action]




class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        #todo 3
        """*** YOUR CODE HERE ***"""
        action, val = self.ab_minimax(game_state, 0, self.depth)
        # print(val)
        # exit(0)
        return action

    def ab_minimax(self, game_state, agent_index, depth, a=-numpy.inf, b=numpy.inf):
        legal_actions = game_state.get_legal_actions(agent_index)
        self.evaluation_function(game_state) #todo for debugging
        if depth == 0 or len(legal_actions) == 0:
            return Action.STOP, self.evaluation_function(game_state)
        actions = {}
        for action in legal_actions:
            state = game_state.generate_successor(agent_index, action)
            minmax_action, value = self.ab_minimax(state, 1 - agent_index, depth - 0.5, a, b)
            actions[action] = value
            if agent_index == 0: #max
                if value >= b:
                    return action, value
                if value > a:
                    a = value
            else: # min
                if value <= a:
                    return action, value
                if value < b:
                    b = value
        if agent_index == 0:
            best_action = max(actions, key=lambda k: actions[k])
            # if depth == 2:
            #     print(f"best action: {best_action}, score given: {actions[best_action]}")
            #     input("enter:")
            return best_action, actions[best_action]
        else:
            best_action = min(actions, key=lambda k: actions[k])
            return best_action, actions[best_action]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # todo 4
        action, val = self.expectimax(game_state, 0, self.depth)
        # print(val)
        # exit(0)
        return action

    def expectimax(self, game_state, agent_index, depth):
        legal_actions = game_state.get_legal_actions(agent_index)
        if depth == 0 or len(legal_actions) == 0:
            return Action.STOP, self.evaluation_function(game_state)
        actions = {}
        for action in legal_actions:
            state = game_state.generate_successor(agent_index, action)
            new_action, value = self.ab_expectimax(state, 1 - agent_index, depth - 0.5)
            actions[action] = value
        if agent_index == 0:
            best_action = max(actions, key=lambda k: actions[k])
            return best_action, actions[best_action]
        else: # enemy choice - choose randomly
            random_action = random.choice(list(actions.keys()))
            avg_value = numpy.average(list(actions.values()))
            return random_action, avg_value







def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION:

    we spent a lot of time trying to find heuristics that work well
    initially we created a linear combination, multiplying by the score as the base.
    with bonuses for having the max tile in the corner,
    for having certain rows monotonically increase to this corner,
    for having more of the board free (either as a sum or a percentage)

    while we sometimes achieved nice results,
    the algorithm could not recover when forced to leave the corner,
    as suddenly the high penalties for corner and monotonicity
    prevented the moves necessary to restore a well balanced board.

    By applying a pattern as the base score, we changed the base multiplier to be a gradient
    where moving higher tiles to the corner always improves their value.
    suddenly the algorithm was able to recover from states even we would have given up on.
    was very impressive to see.

    IDEAS
    - ensure high val in a specific corner
    - montone along sides near this corner
    - bonus for open spaces
    - negative if no move possible
    """
    # todo 5
    s = current_game_state
    # lev = level(s)
    cor = max_in_sw_corner(s)
    # ncor = 1 if (cor == 1) else -1
    occ = num_occupied(s)
    sco = score(s)
    bot = bottom_monotone(s)
    lef = left_monotone(s)
    opn = 1-(occ/16)
    # sid = all_side_monotones(s)
    # dwn = all_down_monotones(s)
    sum_mono = sum_monotones(s)
    pct_mono = sum_mono/8
    pat = pattern_score(s)



    a1 =  sco * (10*cor + 2*lef + 1*bot + opn)     #  20g/depth1: med=3146 avg=3183   10g/depth2: med=4758 avg=6017
    # a1_cond1 = a1 if sco > 50 else cor             #  20g/depth1: med=3882 avg=3817   10g/depth2: med=5574 avg=5690 #todo best so far it seems
    # a1_cond2 = a1 if sco > 50 else cor + lef + bot #  20g/depth1: med=3120 avg=3360   10g/depth2: med=? avg=?
    # a1_cond3 = a1 if sco > 45 else cor             #  20g/depth1: med=3390 avg=3478   10g/depth2: med=? avg=?
    # a1_cond4 = a1 if sco > 40 else cor             #  20g/depth1: med=3272 avg=3406   10g/depth2: med=? avg=?
    # a1_cond5 = a1 if sco > 38 else cor             #  20g/depth1: med=3114 avg=3504   10g/depth2: med=? avg=?
    # a1_cond6 = a1 if sco > 35 else cor             #  20g/depth1: med=? avg=?   10g/depth2: med=? avg=?
    # a2 = sco * (10*cor + 3*lef + 3*bot + opn)     #  20g/depth1: med=2496 avg=3301   10g/depth2: med=? avg=?
    # a3 = sco * (12*cor + 2*lef + 1*bot + opn)     #  20g/depth1: med=2482 avg=2753   10g/depth2: med=? avg=?
    # a4 = sco * (10*cor + 4*lef + 2*bot + opn + 1) #  20g/depth1: med=2086 avg=2603   10g/depth2: med=? avg=?
    # a5 = sco * (10*cor + 4*lef + 2*bot + 5*opn + 1) #  20g/depth1: med=? avg=?   10g/depth2: med=? avg=?
    # a5cond_1 = a5 if sco > 50 else cor  #  20g/depth1: med=? avg=?   10g/depth2: med=? avg=?
    # a6 = sco * (10*cor + 4*lef + 2*bot + opn + 1) #  20g/depth1: med=? avg=?   10g/depth2: med=? avg=?
    # a7 = sco * (10*ncor + 2*lef + 1*bot + 1*opn) #  20g/depth1: med=? avg=?   10g/depth2: med=? avg=?
    # mono1 = sco * (10*cor + 1*lef + 1*bot + 1*dwn + 1*sid + 1*opn) #  20g/depth1: med=? avg=?   10g/depth2: med=? avg=?
    # mono2 = sco * (10*cor + 1*dwn + 1*sid + 1*opn) #  20g/depth1: med=? avg=?   10g/depth2: med=? avg=?
    # mono2_cond = mono2 if sco > 45 else lev*cor    #  20g/depth1: med=? avg=?   10g/depth2: med=? avg=?
    # mono3 = sco * (20*cor + 1*sum_mono+ 4*opn) #  20g/depth1: med=? avg=?   10g/depth2: med=? avg=?  #todo also looking good - achives 7k on more than half!
    # test1 = pat* (20*cor + 1*lef + 1*bot + 1*sum_mono + 4*opn) # 7124 MED 8887 AVG (5/10 7k) 16k max
    # test2 = pat* (5*cor + 1*lef + 1*bot + 1*sum_mono + 2*opn)  # todo  7784 MED 9066 AVG (7/10 7k) 17k max
    # test3 = pat* (5*cor + 2*lef + 1*bot + 1*sum_mono + 2*opn)  # 6924 med 8234 avg (5/10 7k) 14k max
    # test4 = pat* (5*cor + 2*lef + 1*bot + 1*sum_mono + 4*opn)  # meh -  hopefully leaves more open? for endgame meh
    # test5 = pat* (1*cor + 1*lef + 1*bot + 1*pct_mono + 1*opn)  # todo baseline? meh
    # test6 = pat* (7*cor + 1*lef + 1*bot + 1*pct_mono + 2*opn)  # todo  9072 MED 9340 ABG (7/10 7k) 16k max
    test7 = pat* (7*cor + 1*lef + 1*bot + 1*pct_mono + 3*opn) # todo 12570 MED 12986 AVG (9/10 7k) 29k max !!
    # test0 = pat* (7*cor + 1*lef + 1*bot + 1*pct_mono + 4*opn) #  22k max but meh avg no good at all


    return test7

def can_move(current_game_state):
    # todo return false if no legal children
    return current_game_state.generate_successor

# def max_in_sw_corner(current_game_state):
#     corners = current_game_state.board[[0, 0, -1, -1], [0, -1, 0, -1]]
#     return 1 if current_game_state.max_tile == corners[2] else 0

def num_occupied(s):
    return np.count_nonzero(s.board)

def level(s):
    return np.log2(s.max_tile)

def score(s):
    return s.score

def max_in_sw_corner(s):
    # corners = current_game_state.board[[0, 0, -1, -1], [0, -1, 0, -1]]
    return int(s.max_tile == s.board[3][0])

def side_monotone(s, row):
    a = s.board[row][0]
    b = s.board[row][1]
    c = s.board[row][2]
    d = s.board[row][3]

    # a is leftmost should be largest

    return int(a >= b >= c >= d)

def bottom_monotone(s):
    return side_monotone(s, 3)

def down_monotone(s, col):
    a = s.board[0][col]
    b = s.board[1][col]
    c = s.board[2][col]
    d = s.board[3][col]

    # a is highest, should be smallest

    return int(a <= b <= c <= d)

def left_monotone(s):
    return down_monotone(s, 0)

def all_side_monotones(s):
    return all([side_monotone(s, x) for x in range(3)])

def sum_side_monotones(s):
    return sum([side_monotone(s, x) for x in range(3)])

def all_down_monotones(s):
    return all([down_monotone(s, x) for x in range(3)])

def sum_down_monotones(s):
    return sum([down_monotone(s, x) for x in range(3)])

def all_monotones(s):
    return all_down_monotones(s) and all_side_monotones(s)

def sum_monotones(s):
    return sum_down_monotones(s) + sum_side_monotones(s)

def pattern_score(s):
    weights = np.array([[3, 1, 0, 0],
                       [ 5, 3, 1, 0],
                       [ 15,5, 3, 1],
                       [ 30,15,5, 3]])
    return sum(sum(weights*s.board))

# Abbreviation
better = better_evaluation_function
