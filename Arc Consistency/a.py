import random
from timeit import default_timer as timer

file = open('output_019.txt','w')

class minConflict(object):
    def __init__(self, N):
        self.columns, self.leastConflict = [], []
        self.N = N

    def countConflicts(self, column, row):
        count = 0
        for val in xrange(len(self.columns)):
            if val != row:
                nextCol = self.columns[val]
                if nextCol == column or abs(nextCol - column) == abs(val - row):
                    count += 1

        return count


    def initBoard(self):
        for col in xrange(self.N):
            min_conflict = self.N
            self.leastConflict = []
            self.columns.append(0)
            for row in xrange(len(self.columns)):
                print self.leastConflict
                numConflict = self.countConflicts(self.columns[row], col)
                if numConflict == min_conflict:
                    self.leastConflict.append(row)
                elif numConflict < min_conflict:
                    self.leastConflict = []
                    self.leastConflict.append(row)
                    min_conflict = numConflict
            self.columns[col] = random.choice(self.leastConflict)

    def placeQueen(self, queenCol):
        self.leastConflict = []
        min_conflict = 8
        for val in xrange(len(self.columns)):
            conflict_num = self.countConflicts(val, queenCol)
            if conflict_num == min_conflict:
                self.leastConflict.append(val)
            else:
                if conflict_num < min_conflict:
                    self.leastConflict = []
                    min_conflict = conflict_num
                    self.leastConflict.append(val)
        if self.leastConflict:
            self.columns[queenCol] = random.choice(self.leastConflict)

    def nQueen(self):
        moves = 0
        while True:
            self.leastConflict, numConflicts = [], 0
            numConflicts = 0
            for val in range(len(self.columns)):
                numConflicts += self.countConflicts(self.columns[val], val)
            if numConflicts == 0:
                return moves

            queen = random.randint(0, len(self.columns) - 1)
            self.placeQueen(queen)
            self.printBoard()
            moves += 1


    def printBoard(self):
        board = ''
        for row in xrange(len(self.columns)):
            for col in xrange(len(self.columns)):
                if self.columns[col] == row:
                    board += "Q "
                else:
                    board += "+ "
            board += "\n"
        print board
        return board



def main():


    narray = [8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    narray = [8]
    for N in narray:
        print 'N:', N
        file.write('N: ' +str(N) + '\n')
        nqueen = minConflict(N)

        nqueen.initBoard()
        ib = nqueen.printBoard()
        file.write('Initial Board:\n' + ib)


        t = timer()
        moves = nqueen.nQueen()
        t = timer() - t
        board = nqueen.printBoard()
        print "Moves: ", moves
        print "Time:", t, '\n'

        file.write('Final Board: \n' + board + str(moves) + '\n' + str(t) + '\n' + '\n')



if __name__ == "__main__":
    main()
