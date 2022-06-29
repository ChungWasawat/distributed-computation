import os

from math import floor, ceil

import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randn
from numpy.random import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def loss(pos, theta, x, y):
    if pos==0:
        # print(pos, theta[0,pos], y)
        return theta[0,pos] - y
    else:
        # print(pos, theta[0,pos], x[pos-1])
        return loss(pos-1, theta, x, y) + (theta[0,pos] * x[pos-1])

def mean_grad(theta, node, data, X):
    temp_loss = loss()
    for j in range(theta.size):
        if j ==0:
            temps = temp_loss
        else:
            temp = temp_loss * X[data,0] 
            temps = np.c_[temps,temp]  
            
            

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

nrows = X.shape[0]
nodes = 3
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
    datasets.append( shuffle(all_data[ start : stop, : ]) )
    start, stop = stop, stop+divided_n

learning_rate = [0.05, 0.1, 0.5]
epoch = 10
batch_size = 16

old_theta = []
errors = []

seed(99)
theta = randn(1, X.shape[1]+1).flatten()


for t in range(epoch):
    for d in range(max_divided_n):
        pass


