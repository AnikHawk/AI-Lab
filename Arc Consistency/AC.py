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

DOMAIN_VALUE_MIN = 1
DOMAIN_VALUE_MAX = 1000
DOMAIN_SIZE_MIN = 10
DOMAIN_SIZE_MAX = 50
NODES_MIN = 10
NODES_MAX = 50
STEP = 1

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
        r = np.random.choice(range(DOMAIN_VALUE_MIN,DOMAIN_VALUE_MAX), rnd.randint(DOMAIN_SIZE_MIN, DOMAIN_SIZE_MAX), replace=False)
        d.append(r)
    #print(d)
    return d

def add_const(adj,n):
    c = adj.copy()
    for i in range(n):
        for j in range(n):
            c[i][j] *= rnd.randint(1,10)
    return c

def revised(a,b,domain):
    rev = False
    i = 0
    while i < len(domain[a]):
        if i == len(domain[a]): break
        x = constraints[c[a][b]](domain[a][i:i + 1], domain[b][:])
        #print(x)
        if x is False or not any(x):
            rev = True
            domain[a] = np.delete(domain[a], i)
            i -= 1
        i += 1
    return rev, domain

def printDomain(domain):
    for i in range(len(domain)):
        print(domain[i])
    print('\n')

def AC1(edges, domain):
    t = timer()
    rev = 0
    change = True
    while change == True:
        revList = []
        for i in range(len(edges)):
            x, y = edges[i][0], edges[i][1]
            rev += 1
            isRevised , domain = revised(x, y, domain)
            revList.append(isRevised)
        if not any(revList):
            change = False
    return domain, rev, timer() - t


def AC2(edges, domain):
    t = timer()
    rev = 0
    for k in range(n):
        q1 = Queue.Queue()
        q2 = Queue.Queue()
        for e in range(k):
            if adj[k][e] == 1:
                q1.put((k, e))
                q2.put((e, k))
        while not q1.empty():
            while not q1.empty():
                z = q1.get()
                A, B = z[0], z[1]
                rev += 1
                isRevised, domain = revised(A, B, domain)
                if isRevised:
                    for i in range(k):
                        if B != i and adj[i][A] == 1:
                            q2.put((i, A))
            for i in q2.queue: q1.put(i)
            q2 = Queue.Queue()
    return domain, rev, timer() - t

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


def AC4(edges, domain):
    s, counter = {}, {}
    for x in edges:
        a, b = x[0], x[1]
        for xa in domain[a]:
            for xb in domain[b]:
                if constraints[c[a][b]](xa, xb) == True:
                    s[(a, xa)] = s[(a, xa)] + [(b, xb)] if (a, xa) in s else [(b, xb)]
                    counter[(a, xa, b)] = counter[(a, xa, b)] + 1 if (a, xa, b) in counter else 1

    list = collections.deque()
    for x in edges:
        a, b = x[0], x[1]
        for xa in range(len(domain[a])):
            if (a, domain[a][xa], b) not in counter or counter[(a, domain[a][xa], b)] == 0:
                if (a, domain[a][xa]) not in list:
                    list.append((a, domain[a][xa]))

    t = timer()
    removeList = []
    while not len(list) == 0:
        xi, ai = list.pop()
        removeList += [(xi, ai)]
        if (xi, ai) in s:
            for j in s[(xi, ai)]:
                xj, aj = j[0], j[1]
                if (xj, aj, xi) in counter:
                    counter[(xj, aj, xi)] -= 1
                    if counter[(xj, aj, xi)] == 0:
                        list.append((xj, aj))
    return removeList, timer() - t


t1, t2, t3, t4 = [],[],[],[]
r1, r2, r3, r4 = [],[],[],[]


x = []

for i in range(NODES_MIN,NODES_MAX+1,STEP):

    #n = rnd.randint(20,50)
    n = i
    x += [n]
    print(n)
    adj = gen_adj_matrix(n,.1)
    d = gen_domain(n)
    c = add_const(adj, n)
    edges = zip(*np.where(adj == 1))

    domain = d[:]
    #printDomain(domain)

    domain, r, t = AC1(edges,domain)
    t1 += [t]
    #printDomain(domain)
    #print '\n'

    domain = d[:]
    domain, r, t = AC2(edges, domain)
    t2 += [t]
    #printDomain(domain)
    #print '\n'

    domain = d[:]
    domain, r, t = AC3(edges,domain)
    t3 += [t]
    #printDomain(domain)
    #print '\n'

    domain = d[:]
    rmv, t = AC4(edges, domain)
    t4 += [t]

#print np.sum(r1),'\n',np.sum(r2),'\n', np.sum(r3)

print np.sum(t1),'\n',np.sum(t2), '\n', np.sum(t3), '\n', np.sum(t4)

# plt.plot(x,t1)
# plt.plot(x,t2)
# plt.plot(x,t3)
# plt.plot(x,t4)
# plt.legend(['AC1', 'AC2', 'AC3', 'AC4'], loc='upper left')
#
# plt.show()


xnew = np.linspace(NODES_MIN,NODES_MAX,1000) #300 represents number of points to make between T.min and T.max
t1new = spline(x,t1,xnew)
t2new = spline(x,t2,xnew)
t3new = spline(x,t3,xnew)
t4new = spline(x,t4,xnew)


plt.plot(xnew,t1new)
plt.plot(xnew,t2new)
plt.plot(xnew,t3new)
plt.plot(xnew,t4new)
plt.legend(['AC1', 'AC2', 'AC3', 'AC4'], loc='upper left')

plt.show()





#
# G = nx.from_numpy_matrix(np.array(adj))
# nx.draw(G, with_labels=True)
# plt.show()
#

