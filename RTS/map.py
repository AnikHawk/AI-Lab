TRACK = """
        ##################
        ####             #
        ###              #
        ###              x
        ##               #
        #    +           #
        #                #
        # +        #######
        #         ########
        #       + ########
        #         ########
        #         ########
        # +       ########
        #         ########
        #         ########
        ##   o    ########
        __________________
        """

TRACK = [list(x) for x in TRACK.split("\n") if x]

# Define cost of moving around the map
cost_regular = 1.0
cost_diagonal = 1.0

# Create the cost dictionary
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

class RTS():
    def __init__(self, board):
        self.board = board
        self.goal = (0, 0)

        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == "o":
                    self.initial = (x, y, 0, 0)
                elif self.board[y][x].lower() == "x":
                    self.goal = (x,y)