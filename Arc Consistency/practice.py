import datetime
from scipy.interpolate import spline
import collections
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy.random as rnd
import itertools
import Queue
from timeit import default_timer as timer
import copy

def c0(x,y):
    return False
def c1(x,y):
    return x < y
def c2(x,y):
    return x > y
def c3(x,y):
    return x <= 2*y
def c4(x,y):
    return x <= 3*y
def c5(x,y):
    return x > y**2
def c6(x,y):
    return x <= y**2
def c7(x,y):
    return x + y > 500
def c8(x,y):
    return x + y < 500
def c9(x,y):
    return 2*x + 3*y < 1000
def c10(x,y):
    return x == y
constraints = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]



def gen_adj_matrix(n, e):
    adj = np.random.randint(1, 100, (n, n))
    adj = (adj + adj.T) / 2
    adj[adj <= e*100] = 1
    adj[adj > e*100] = 0
    # print(adj)
    return adj


def gen_domain(n):
    d = []
    for i in range(n):
        r = np.random.choice(range(1,1000), rnd.randint(10, 50), replace=False)
        d.append(r)
    #print(d)
    return d

def add_const(adj,n):
    c = adj.copy()
    for i in range(n):
        for j in range(n):
            c[i][j] *= rnd.randint(1,10)
    return c
#
# def revised(a,b,domain):
#     isRev = False
#     i = 0
#     while i < len(domain[a]):
#         if i == len(domain[a]): break
#         x = constraint(domain[a][i:i + 1], domain[b][:])
#         print(x)
#
#         if x is False or not any(x):
#             isRev = True
#             domain[a] = np.delete(domain[a], i)
#             i -= 1
#         i += 1
#     return isRev, domain

def forwardChecking(domain):
    dmn = domain[:]
    q = Queue.Queue()
    assingment = []
    for i in range(n):
        q.put(i)
    while not q.empty():
        x = q.get()
        for xi in dmn[x]:
            dmn = checkForward(x,xi,dmn)
            if dmn == False: return False
            else:
                assingment.append((x,xi))
                break
    return assingment


def checkForward(x, xi, domain):
    dmn = domain[:]
    for y in range(n):
        if(adj[x][y] == 1):
            for yi in dmn[y]:
                if constraint(xi,yi) == False:
                    dmn[y].remove(yi)
                    if len(dmn[y]) == 0: return False

    return dmn



def revised(a,b,domain):
    isRev = False
    for ai in domain[a]:
        x=[]
        for bi in domain[b]:
            x.append(constraint(ai,bi))
        if not any(x):
            isRev = True
            domain[a].remove(ai)
    return isRev,domain


def printDomain(domain):
    for i in range(len(domain)):
        print(domain[i])
    print('\n')

def AC3(edges, domain):
    t = timer()
    rev = 0
    q = Queue.Queue()
    for i in range(len(edges)): q.put(edges[i])
    while not q.empty():
        e = q.get()
        x, y = e[0], e[1]
        rev += 1
        isRevised, domain = revised(x, y, domain)
        if isRevised:
            for i in range(n):
                if i != x and i != y and adj[i][x] == 1:
                    q.put((i,x))
    return domain, rev, timer() - t

def isConsistent(x,xi,domain):
    for y in range(n):
        if(adj[x][y] == 1):
            c = []
            for yi in domain[y]:
                c.append(constraint(xi,yi))
            if not any(c):
                return False
    return True

def backtrackingsearch(assignment, unassignedVar, edges, domain):
    if len(assignment) == n: return assignment
    var = unassignedVar.get()
    for xi in domain[var]:
        if isConsistent(var,xi,domain):
            checkForward(var, xi, domain)
            assignment.append((var,xi))
            result = backtrackingsearch(assignment, unassignedVar, edges, domain)
            return result
            assignment.remove((var,xi))
    return False






assingment = []
n = 3
unassignedVar = Queue.Queue()
for i in range(n):
    unassignedVar.put(i)


adj = np.array([[0,1,1],
               [1,0,1],
               [1,1,0]])
edges = zip(*np.where(adj == 1))
domain = [['r','g','b'],['g'], ['r']]
#domain = [[3],[1,2,3],[1,3]]
def constraint(x,y):
    return x != y



assingment = backtrackingsearch([],unassignedVar,edges,domain)
print(assingment)


a, i = 2, 1
#print(isConsistent(2,3,domain))
#domain = forwardChecking(domain)
#domain, isRev, t = AC3(edges,domain)
#isRev, domain = revised(0,2,domain)
#print(domain)









#
# G = nx.from_numpy_matrix(np.array(adj))
# nx.draw(G, with_labels=True)
# plt.show()
#

