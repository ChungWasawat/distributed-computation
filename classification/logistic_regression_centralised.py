import os

from math import floor, ceil

import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randn
from numpy.random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def theta_init(seed_num):
    seed(seed_num)
    theta = randn(1, X.shape[1]+1).flatten()
    return theta[0], theta[1:]

def sigmoid(w, X, b):
    X_T = X.reshape(X.size,1)
    xx = np.dot(w.reshape(1, w.size), X_T).flatten() + b
    return 1 / (1 + np.exp(-xx))

def compute_grad(a,y,X_T):
    loss = a-y
    grad = loss * X_T
    return grad.flatten()

def cost(a,y):
    return (-y * np.log10(a)) - ((1-y) * np.log10(1-a))

##########################################################
"""
1. variance of Wavelet Transformed image (continuous)
2. skewness of Wavelet Transformed image (continuous)
3. curtosis of Wavelet Transformed image (continuous)
4. entropy of image (continuous)
5. class (integer)
"""

path = os.getcwd()

data1 = path + "\data\\data_banknote_authentication.txt"
col1 = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

df1 = pd.read_table(data1, sep=",", names=col1)

X = df1.values[:, 0:2] 
Y = df1.values[:, 4]

std_scaler = StandardScaler()
X = std_scaler.fit_transform(X)

##########################################################
# logistic regression
model = LogisticRegression()
model.fit(X, Y)

print("from logistic regression", model.intercept_.round(decimals=3), model.coef_.round(decimals=3))

col_table = ["model", "learning rate", "batch", "epoch", "intercept"]
table = [["logistic regression", 0, 0, 0, model.intercept_[0].round(decimals=3)]]
for m in range(model.coef_.size):
    table[0].append(model.coef_.flatten()[m].round(decimals=3))
    col_table.append(f"theta{m+1}")
    
##########################################################
## sgd
theta_is_zero = True
seed_num = 99

learning_rate = [0.005, 0.01, 0.05, 0.1]
# learning_rate = [0.01, 0.05, 0.1, 0.5]

epoch = 10
every_t = 1
batch_size = 128

all_data = np.c_[X,Y]
nodes = [3,4,5]

for node in nodes:
    nrows = all_data.shape[0]
    divided_n = floor( nrows/node)
    max_divided_n = ceil( nrows/node) 
    remain_d = nrows%node
    
    datasets = []
    start, stop = 0, divided_n
    # divide the entire dataset to n nodes
    for n in range(node):   
        if remain_d > 0:
            stop+=1
            remain_d -= 1
        # shuffle for sgd
        seed(seed_num)
        shuffle(all_data[ start : stop, : ])
        datasets.append( all_data[ start : stop, : ] )
        start, stop = stop, stop+divided_n


















