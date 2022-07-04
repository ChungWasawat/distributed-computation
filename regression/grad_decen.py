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

def random_network()