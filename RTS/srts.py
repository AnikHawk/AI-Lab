import math
from simpleai.search import SearchProblem, greedy


class SRTS(SearchProblem):
    def __init__(self, board):
        self.board = board
        self.goal = (0, 0)

        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == "o":
                    self.initial = (x, y)
                elif self.board[y][x].lower() == "x":
                    self.goal = (x, y)

        super(SRTS, self).__init__(initial_state=self.initial)


    def actions(self, state):
        actions = []
        for action in COSTS.keys():
            newx, newy = self.result(state, action)
            if self.board[newy][newx] != "#" and self.board[newy][newx] != "_":
                actions.append(action)

        return actions

    def result(self, state, action):
        x, y = state
        if action.count("up"):
            y -= 1
        if action.count("down"):
            y += 1
        if action.count("left"):
            x -= 1
        if action.count("right"):
            x += 1
        new_state = (x, y)
        return new_state

    def is_goal(self, state):
        return state == self.goal

    def is_safe(self, state):
        x, y = state
        return self.board[x][y] == '+' or self.board[x][y] == 'x'

    def cost(self, state, action, state2):
        return COSTS[action]

    def heuristic(self, state):
        x, y = state
        gx, gy = self.goal

        safe_dist = float('inf')
        for by in range(len(self.board)):
            for bx in range(len(self.board[by])):
                if self.board[by][bx] == "+":
                    safe_dist = \
                        min(math.sqrt((x - bx) ** 2 + (y - by) ** 2),safe_dist)

        if safe_dist == 0: safe_dist = 2
        goal_dist = math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

        return min(safe_dist,goal_dist)

from simpleai.search import greedy as best_first
if __name__ == "__main__":
    TRACK = """
        ###################################
        ###################################
        ######   +                     + ##
        ####                             ##
        ###                               x
        ##   +                           ##
        ##             +               + ##
        ##           ######################
        ##          #######################
        ## +       ########################
        ##         ########################
        ##         ########################
        ##       + ########################
        ##         ########################
        ##         ########################
        ##    o    ########################
        ___________________________________
        """

    print(TRACK)
    TRACK = [list(x) for x in TRACK.split("\n") if x]
    result = None
    cost_regular = 1.0
    cost_diagonal = math.sqrt(2)

    COSTS = {
        "up": cost_regular,
        "down": cost_regular,
        "left": cost_regular,
        "right": cost_regular,
        "up left": cost_diagonal,
        "up right": cost_diagonal,
        "down left": cost_diagonal,
        "down right": cost_diagonal,
    }

    problem = SRTS(TRACK)
    s_root = problem.initial
    s_goal = problem.goal

    while s_root != s_goal:
        Comfortables = []
        c = best_first(problem, graph_search=True)
        t = s_root
        if c is not None or problem.is_safe(c):
            Comfortables.append(t)
        if len(Comfortables) is not 0:
            result = greedy(problem, graph_search=True)
            break

        ancestors = [a[1] for a in c.path()]
        s_safe = Comfortables[0]
        s_target = Comfortables[0]
        for a in ancestors:
            Comfortables.append(a)

        if problem.is_safe(Comfortables[0]):
            s_target = s_safe

        elif problem.is_safe(problem.result(s_target,'up')) or \
            problem.is_safe(problem.result(s_target, 'up')) or \
            problem.is_safe(problem.result(s_target, 'up')) or \
            problem.is_safe(problem.result(s_target, 'up')) or \
            problem.is_safe(problem.result(s_target, 'up')) or \
            problem.is_safe(problem.result(s_target, 'up')) or \
            problem.is_safe(problem.result(s_target, 'up')) or \
            problem.is_safe(problem.result(s_target, 'up')):
                s_target = s_safe
        else:
            s_target = s_root
            print('No Solution')
            break

        problem.initial = s_root
        problem.goal = s_target
        s_root = s_target


    path = [x[1] for x in result.path()]
    print()
    for y in range(len(TRACK)):
        for x in range(len(TRACK[y])):
            if (x, y) == problem.initial:
                print('o', end='')
            elif (x, y) == problem.goal:
                print('x', end='')
            elif (x, y) in path:
                print('·', end='')
            else:
                print(TRACK[y][x], end='')

        print()


