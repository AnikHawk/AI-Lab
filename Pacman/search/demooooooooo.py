

import util

def graphSearch(problem, frontier):

    explored = []
    frontier.push([(problem.getStartState(), "Stop", 0)])

    while not frontier.isEmpty():
        # print "frontier: ", frontier.heap
        path = frontier.pop()
        # print "path len: ", len(path)
        # print "path: ", path

        s = path[len(path) - 1]
        s = s[0]
        # print "s: ", s
        if problem.isGoalState(s):
            # print "FOUND SOLUTION: ", [x[1] for x in path]
            return [x[1] for x in path][1:]

        if s not in explored:
            explored.append(s)
            # print "EXPLORING: ", s

            for successor in problem.getSuccessors(s):
                # print "SUCCESSOR: ", successor
                if successor[0] not in explored:
                    successorPath = path[:]
                    successorPath.append(successor)
                    # print "successorPath: ", successorPath
                    frontier.push(successorPath)
            # else:
            # print successor[0], " IS ALREADY EXPLORED!!"


    return []


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"
    cost = lambda aPath: problem.getCostOfActions([x[1] for x in aPath])
    frontier = util.PriorityQueueWithFunction(cost)
    return graphSearch(problem, frontier)
