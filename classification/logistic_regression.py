import os

from math import floor, ceil

import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randn
from numpy.random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def theta_init():
    seed(99)
    theta = randn(1, X.shape[1]+1).flatten()
    return theta[0], theta[1:]

def sigmoid(w, X_T):
    xx = np.dot(w, X_T)
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

# std_scaler = StandardScaler()
# X = std_scaler.fit_transform(X)

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
# sgd
learning_rate = [0.005, 0.01, 0.05, 0.1]
epoch = 1
batch_size = 32

old_theta = []
errors = []

all_data = np.c_[X,Y]
nrows= all_data.shape[0]

for lr in learning_rate:   
    theta0, theta = theta_init()

    old_th = []
    error = []    

    for t in range(epoch):
        for m in range(nrows):
            x = X[0].reshape(X[0].size,1)
            y = Y[0]
            a = sigmoid(theta, x)
            c = cost(a,y)
            g = compute_grad(a, y, x)

            theta0 = theta0 - (lr * (a-y))
            theta = theta - (lr * g)
            break
    break