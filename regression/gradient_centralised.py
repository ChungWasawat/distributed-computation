import os

from math import floor, ceil

import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randn
from numpy.random import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def loss(pos, theta, x, y):
    if pos==0:
        return theta[pos] - y
    else:
        return loss(pos-1, theta, x, y) + (theta[pos] * x[pos-1])

def all_grad(theta, node, order, dataset):  
    if node ==0:
        try:
            temp_loss = loss(theta.size-1, theta, dataset[node][order][0:-1], dataset[node][order][-1])
            for j in range(theta.size):
                if j ==0:
                    temp_grad = temp_loss
                else:
                    t = temp_loss * dataset[node][order][j-1]
                    temp_grad = np.c_[temp_grad,t] 
            return ( temp_grad, 1 )
        except:
            return ( np.zeros((1,theta.size)), 0 )
    else:
        try:
            grad, ne = all_grad(theta, node-1, order, dataset) # ne for dataset is Not Equal
            temp_loss = loss(theta.size-1, theta, dataset[node][order][0:-1], dataset[node][order][-1])
            for j in range(theta.size):
                if j ==0:
                    temp_grad = temp_loss
                else:
                    t = temp_loss * dataset[node][order][j-1]
                    temp_grad = np.c_[temp_grad,t] 
            return ( grad + temp_grad, ne+1)
        except:
            return ( grad, ne)
      
def contour(m, path, lr):
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
    title = f"learning rate = {lr}"
    plt.title(title)
    plt.xlabel("theta1")
    plt.ylabel("theta2")
    plt.show()            

path = os.getcwd()

data1 = path + "\data\\airfoil_self_noise.dat"
col1 = ['freq', 'angle', 'chord', 'velocity', 'thickness', 'sound']
df1 = pd.read_table(data1, sep="\t", names=col1)

X = df1.values[:, 0::3] #freq and velocity
# X = df1.values[:, 0:5] 
y = df1.values[:, 5]

std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)
all_data = np.c_[X,y]

##########################################################
# linear regression
model = LinearRegression()
model.fit(X, y)

print("from linear regression", model.intercept_.round(decimals=3), model.coef_.round(decimals=3))

##########################################################
# sgd
nrows = all_data.shape[0]
nodes = 5
divided_n = floor( nrows/nodes)
max_divided_n = ceil( nrows/nodes) 
remain_d = nrows%nodes


datasets = []
start, stop = 0, divided_n
# divide the entire dataset to n nodes
for n in range(nodes):   
    if remain_d > 0:
        stop+=1
        remain_d -= 1
    # shuffle for sgd
    seed(99)
    shuffle(all_data[ start : stop, : ])
    datasets.append( all_data[ start : stop, : ] )
    start, stop = stop, stop+divided_n

learning_rate = [0.005, 0.01, 0.05, 0.1]
epoch = 5
batch_size = 16

old_theta = []
errors = []

for lr in learning_rate:
    seed(99)
    theta = randn(1, X.shape[1]+1).flatten()

    old_th = []
    for t in range(epoch):
        for d in range(max_divided_n):
            mean_grad, ne = all_grad(theta, nodes-1, d, datasets) 
            mean_grad /= ne
            if d % 50 ==0:
                old_th.append(theta[1:])
            theta = theta - lr * mean_grad.flatten()


    print("learning rate=", lr, theta.round(decimals=3))
    old_theta.append(old_th)   
 

for lr in range(len(learning_rate)):
    all_w = np.array(old_theta[lr])
    if all_w.shape[1] > 2:
        break
    contour(model.coef_, all_w, learning_rate[lr])  

