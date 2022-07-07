import os

from math import floor, ceil

import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randn
from numpy.random import uniform
from numpy.random import shuffle
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import StandardScaler

#import matplotlib.pyplot as plt

def theta_init(seed_num: int, row: int) -> np.array:
    seed(seed_num)
    theta = randn(1, row).flatten()
    return theta[0], theta[1:]

def y_hat(w: np.array, X: np.array, b: np.array) -> np.array:
    X_T = X.reshape(X.size,1)
    xx = np.dot(w.reshape(1, w.size), X_T).flatten() + b
    return xx

def compute_grad(y_hat,y ,X_T):
    loss = y_hat-y
    grad = loss * X_T
    return grad.flatten()

def contour(m, path, lr, node):
    plt.figure()
    x,y = m[0], m[1]
    plt.xlim(-x*3, x*3)
    plt.ylim(-y*7, y*7)
    plt.grid()
    
    for i in range(path.shape[0]-1):
        if i == 0:
            plt.annotate('', xy=path[i + 1, :], xytext=path[i, :],
                         arrowprops={'arrowstyle': '->', 'color': 'green', 'lw': 1},
                         va='center', ha='center')
        else:
            plt.annotate('', xy=path[i + 1, :], xytext=path[i, :],
                         arrowprops={'arrowstyle': '->', 'color': 'red', 'lw': 1},
                         va='center', ha='center')            
        
    plt.plot(x, y, marker="o", markersize=10, markeredgecolor="blue", markerfacecolor="none", zorder=10)
    title = f"learning rate = {lr}, node = {node}"
    plt.title(title)
    plt.xlabel("theta1")
    plt.ylabel("theta2")
    plt.show()

def converge(error, step, lr):
    plt.figure()
    plt.xlim(0, step)
    plt.plot(error, color = 'b')
    title = f"learning rate = {lr}"
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("error")
    plt.show()

def random_network(node: int, p: float) -> np.array:
    nw = np.zeros((node,node))

    route = {}
    for i in range(node):
        route[i] = []

    c = (node **2)
    for i in range(node-1,0,-1):
        c -= i

    i,j = 0,0
    while c > 0:
        if i==j:
            nw[i,j] = 1
            nw[j,i] = 1
        else:
            x = uniform(0,1)
            if x <= p:
                nw[i,j] = 1
                nw[j,i] = 1 
                if j not in route[i]:
                    route[i].append(j)  
                    route[j].append(i)
        j+=1
        if j == node:
            i += 1
            j = i
        c -=1
    return nw, route

def visit_adj(adj_m, node, neighbour, visited):
    if adj_m[node] == []:
        print("case0", node, neighbour, "nothing", visited)
        return visit_adj(adj_m, node+1, neighbour, visited) 
    elif adj_m[node] == [] and node == len(adj_m.keys())-1:
        print("case0.5", node, neighbour, "nothing", visited)
        return visited 
    if visited[node] == 0:
        visited[node] = 1
    if visited[adj_m[node][neighbour]] ==1 and adj_m[node][neighbour] == adj_m[node][-1]:
        print("case1", node, neighbour, adj_m[node][neighbour], visited)
        return visited
    elif visited[adj_m[node][neighbour]] ==1 :
        print("case2", node, neighbour, adj_m[node][neighbour], visited)
        return visit_adj(adj_m, node, neighbour+1, visited) ##
    elif visited[adj_m[node][neighbour]] ==0 and adj_m[node][neighbour] == adj_m[node][-1]:
        print("case3", node, neighbour, adj_m[node][neighbour], visited)
        return visit_adj(adj_m, adj_m[node][neighbour], 0, visited)
    else:
        print("case4", node, neighbour, adj_m[node][neighbour], visited)
        visited = visit_adj(adj_m, adj_m[node][neighbour], 0, visited) ##
        return visit_adj(adj_m, node, neighbour+1, visited)







a,b = random_network(5, 0.5)
b = {0: [1], 1: [0, 2, 4], 2: [1, 4], 3: [], 4: [1, 2]}
#print(a, b)
print(b)

#for c in b.keys():
#    print(c, group_adj(b, c, 0))

for c in b.keys():
    print(c, visit_adj(0, b[c], []))

print("new", visit_adj2(b, 0, 0, [0]*5))
