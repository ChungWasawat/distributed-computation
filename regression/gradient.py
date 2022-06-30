import os
import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randn
from numpy.random import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


def gradient(pos, theta, x, y):
    if pos==0:
        # print(pos, theta[0,pos], y)
        return theta[0,pos] - y
    else:
        # print(pos, theta[0,pos], x[pos-1])
        return gradient(pos-1, theta, x, y) + (theta[0,pos] * x[pos-1])
  
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
    
def converge(error, step, lr):
    plt.figure()
    plt.xlim(0, step)
    plt.plot(error, color = 'b')
    title = f"learning rate = {lr}"
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("error")
    plt.show()
    
"""
Data1 has the following inputs:
1. Frequency, in Hertzs.
2. Angle of attack, in degrees.
3. Chord length, in meters.
4. Free-stream velocity, in meters per second.
5. Suction side displacement thickness, in meters.

The only output is:
6. Scaled sound pressure level, in decibels.
"""

path = os.getcwd()

data1 = path + "\data\\airfoil_self_noise.dat"
col1 = ['freq', 'angle', 'chord', 'velocity', 'thickness', 'sound']
df1 = pd.read_table(data1, sep="\t", names=col1)

# X = df1.values[:, 0::3] #freq and velocity
X = df1.values[:, 0:5] 
y = df1.values[:, 5]

std_scaler = StandardScaler()

X = std_scaler.fit_transform(X)

##########################################################
###### linear regression 

model = LinearRegression()
model.fit(X, y)

print("from linear regression", model.intercept_.round(decimals=3), model.coef_.round(decimals=3))

##########################################################
###### stochastic gradient descent

learning_rate = [0.005, 0.01, 0.05, 0.1]
epoch =10
batch_size = 128
nrows = X.shape[0]

all_data = np.c_[X,y]

old_theta = []
errors = []

for lr in learning_rate:
    seed(99)
    theta = randn(1, X.shape[1]+1)
    old_th = []
    old_th.append(theta[:,1:].reshape(theta.size-1,))
    error = []
    
    shuffle(all_data)
    train_x = all_data[:, :-1]
    train_y = all_data[:, -1]  
  
    for i in range(epoch):
        for start in range(0, nrows):
            loss = gradient(theta.size-1, theta, train_x[start], train_y[start])
            
            for j in range(theta.size):
                if j ==0:
                    temps = theta[0,j] - (lr * loss)
                else:
                    temp = theta[0,j] - (lr * loss * train_x[start,j-1] )
                    temps = np.c_[temps,temp]        
            theta = temps
            
            # store only for a start point or every batch
            if start == 0 or (start+1) % batch_size == 0:
                old_th.append(theta[:,1:].reshape(theta.size-1,))
            # store only for the first iteration.
            if i ==0:
                error.append(abs(loss))
                
    print("learning rate=", lr, theta.flatten().round(decimals=3))
    old_theta.append(old_th)    
    errors.append(error)

# visualisation
## gradient
for lr in range(len(learning_rate)):
    all_w = np.array(old_theta[lr])
    if all_w.shape[1] > 2:
        break
    contour(model.coef_, all_w, learning_rate[lr])    
## error
for lr in range(len(learning_rate)):
    all_e = np.array(errors[lr])
    converge(all_e, len(errors[lr]), learning_rate[lr])
    

##########################################################
####### stochastic gradient descent (mini-batch)

old_theta = []
errors = []

for lr in learning_rate:
    seed(99)
    theta = randn(1, X.shape[1]+1)
    sum_grad = np.zeros(theta.shape)
    
    old_th = []
    old_th.append(theta[:,1:].reshape(theta.size-1,))
    error = []
    
    shuffle(all_data)
    train_x = all_data[:, :-1]
    train_y = all_data[:, -1]
    
    for i in range(epoch):
        for start in range(0, nrows):
            #stop = start + batch_size
            loss = gradient(theta.size-1, theta, train_x[start], train_y[start])
                 
            # for j in range(theta.size):
            #     if j ==0:
            #         temps = theta[0,j] - (lr * loss)
            #     else:
            #         temp = theta[0,j] - (lr * loss * train_x[start,j-1] )
            #         temps = np.c_[temps,temp]  
            for j in range(theta.size):
                if j ==0:
                    temps = loss
                else:
                    temp = loss * train_x[start,j-1] 
                    temps = np.c_[temps,temp]              
            sum_grad += temps
            
            if (start+1) % batch_size == 0 or start == nrows -1: 
                if start == nrows-1:
                    
                    sum_grad /= (nrows%batch_size)
                else:
                    sum_grad /= batch_size
                theta = theta - (lr * sum_grad)
                sum_grad = np.zeros(theta.shape)
                
                old_th.append(theta[:,1:].reshape(theta.size-1,)) 
                error.append(abs(loss))
               
    print("learning rate=", lr, theta.flatten().round(decimals=3))
    old_theta.append(old_th)    
    errors.append(error)


# visualisation
## gradient
for lr in range(len(learning_rate)):
    all_w = np.array(old_theta[lr])
    if all_w.shape[1] > 2:
        break
    contour(model.coef_, all_w, learning_rate[lr])    
## error
for lr in range(len(learning_rate)):
    all_e = np.array(errors[lr])
    converge(all_e, len(errors[lr]), learning_rate[lr])

