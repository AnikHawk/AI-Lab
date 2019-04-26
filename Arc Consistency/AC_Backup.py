from scipy.interpolate import spline
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy.random as rnd
from timeit import default_timer as timer

DOMAIN_VALUE_MIN = 50
DOMAIN_VALUE_MAX = 1000
DOMAIN_SIZE_MIN = 20
DOMAIN_SIZE_MAX = 30
NODES_MIN = 50
NODES_MAX = 250
STEP = 50
THRESHOLD = .5
M = 10

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
        r = np.random.choice(range(DOMAIN_VALUE_MIN,DOMAIN_VALUE_MAX),
                             rnd.randint(DOMAIN_SIZE_MIN, DOMAIN_SIZE_MAX), replace=False)
        d.append(r)
    return d

def add_const(adj,n):
    c = adj.copy()
    for i in range(n):
        for j in range(n): c[i][j] *= rnd.randint(1,11)
    return c

def revised(a,b,domain):
    rev = False
    i = 0
    while i < len(domain[a]):
        if i == len(domain[a]): break
        x = constraints[c[a][b]](domain[a][i:i + 1], domain[b][:])
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
        q1, q2 = [], []
        for e in range(k):
            if adj[k][e] == 1:
                q1.append((k, e))
                q2.append((e, k))
        while q1:
            while q1:
                A, B = q1.pop()
                rev += 1
                isRevised, domain = revised(A, B, domain)
                if isRevised:
                    for i in range(k):
                        if B != i and adj[i][A] == 1:
                            list(set().union(q2, [(i, A)]))
            q1 = q2
            q2 = []
    return domain, rev, timer() - t

def AC3(edges, domain):
    t = timer()
    rev = 0
    q = []
    for i in range(len(edges)): q.append(edges[i])
    while q:
        x, y = q.pop()
        rev += 1
        isRevised, domain = revised(x, y, domain)
        if isRevised:
            if len(domain[x]) == 0: return domain, rev, timer() - t
            for i in range(n):
                if i != x and i != y and adj[i][x] == 1:
                    list(set().union(q, [(i,x)]))
    return domain, rev, timer() - t

def AC4(edges, domain):
    t = timer()
    s, counter, lst = {}, {}, []
    for x in range(len(domain)):
        for xi in domain[x]:
            for y in range(len(domain)):
                counter[(x,xi,y)] = 0

    for x in range(len(domain)):
        for xi in domain[x]:
            s[(x,xi)] = []

    for x in edges:
        a, b = x[0], x[1]
        for xa in domain[a]:
            for xb in domain[b]:
                if constraints[c[a][b]](xa, xb) == True:
                    list(set().union(s[(a,xa)], [(b,xb)]))
                    counter[(a, xa, b)] += 1
            if counter[(a,xa,b)] == 0:
                list(set().union(lst, [(a, xa)]))
                idx = np.where(domain[a]==xa)
                domain[a] = np.delete(domain[a], idx)
                if len(domain[a] == 0): return [],timer() -t


    removeList = []
    while lst:
        xi, ai = lst.pop()
        removeList += [(xi, ai)]
        if (xi, ai) in s:
            for j in s[(xi, ai)]:
                xj, aj = j[0], j[1]
                if (xj, aj, xi) in counter:
                    counter[(xj, aj, xi)] -= 1
                    if counter[(xj, aj, xi)] == 0:
                        list(set().union(lst, [(xj,aj)]))
    return removeList, timer() - t


t1, t2, t3, t4 = [],[],[],[]
r1, r2, r3, r4 = [],[],[],[]
x = []
for i in range(NODES_MIN,NODES_MAX+1,STEP):
    #rnd.seed(11)
    n = i
    x += [n]
    #print(n)
    tt1,tt2,tt3,tt4 = [],[],[],[]
    for j in range(M):
        adj = gen_adj_matrix(n,THRESHOLD)
        d = gen_domain(n)
        c = add_const(adj, n)
        edges = zip(*np.where(adj == 1))
        domain = d[:]
        domain, r, t = AC1(edges, domain)
        tt1.append(t)

        domain = d[:]
        domain, r, t = AC2(edges, domain)
        tt2.append(t + np.sum(t2))

        domain = d[:]
        domain, r, t = AC3(edges, domain)
        tt3.append(t + np.sum(t3))

        domain = d[:]
        rmv, t = AC4(edges, domain)
        tt4.append(t)

    t1.append(np.average(tt1))
    t2.append(np.average(tt2))
    t3.append(np.average(tt3))
    t4.append(np.average(tt4))


#print np.sum(r1),'\n',np.sum(r2),'\n', np.sum(r3)
print np.sum(t1),'\n',np.sum(t2), '\n', np.sum(t3), '\n', np.sum(t4)

print('\n')
for i in t1: print(i)
print('\n')

for i in t2: print(i)
print('\n')

for i in t3: print(i)
print('\n')

for i in t4: print(i)
print('\n')



plt.plot(x,t1)
plt.plot(x,t2)
plt.plot(x,t3)
plt.plot(x,t4)
plt.show()
#
# fig = plt.figure(1)
# plt.subplot(121)
# xnew = np.linspace(NODES_MIN,NODES_MAX,500)
# r1new = spline(x,r1,xnew)
# r2new = spline(x,r2,xnew)
# r3new = spline(x,r3,xnew)
# plt.plot(xnew,r1new)
# plt.plot(xnew,r2new)
# plt.plot(xnew,r3new)
# plt.legend(['AC1', 'AC2', 'AC3'], loc='upper left')
# plt.xlabel('Number of Nodes (n)', fontsize=10)
# plt.ylabel('Times Revised (r)', fontsize=10)
# #plt.show()
# xnew = np.linspace(NODES_MIN,NODES_MAX,1000)
# t1new = spline(x,t1,xnew)
# t2new = spline(x,t2,xnew)
# t3new = spline(x,t3,xnew)
# t4new = spline(x,t4,xnew)
# plt.subplot(122)
# plt.plot(xnew,t1new)
# plt.plot(xnew,t2new)
# plt.plot(xnew,t3new)
# plt.plot(xnew,t4new)
# plt.legend(['AC1', 'AC2', 'AC3', 'AC4'], loc='upper left')
# plt.xlabel('Number of Nodes (n)', fontsize=10)
# plt.ylabel('Runtime (s)', fontsize=10)
# plt.show()
# fig.savefig('graph.png')