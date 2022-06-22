import os

import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randn
from numpy.random import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def hypothesis(X, theta):
    return np.dot(X, theta)
def gradient(X, y, theta):
    h = hypothesis(X, theta)
    grad = np.dot(X.transpose(), (h - y))
    return grad
def cost(X, y, theta):
    h = hypothesis(X, theta)
    J = np.dot((h - y).transpose(), (h - y))
    J /= 2
    return J[0]



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

X = df1.values[:, 0::3] 
y = df1.values[:, 5]

std_scaler = StandardScaler()

X = std_scaler.fit_transform(X)

###### linear regression 

model = LinearRegression()
model.fit(X, y)
print(model.intercept_, model.coef_)

###### gradient descent

learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1]
epoch = 10
batch_size = 128
nrows = 1000 #X.shape[0]

all_data = np.c_[X,y]

for lr in learning_rate:
    seed(47)
    theta = randn(1, X.shape[1]+1)
    for i in range(epoch):
        #shuffle(all_data)
        train_x = all_data[:, :-1]
        train_y = all_data[:, -1]

        for start in range(0, nrows):
            #stop = start + batch_size
            grad = theta[0,0] + (theta[0,1]*train_x[start,0]) + (theta[0,2]*train_x[start,1]) - train_y[start]
            for j in range(theta.size):
                if j ==0:
                    temps = theta[0,j] - (lr * grad)
                else:
                    temp = theta[0,j] - (lr * grad * train_x[start,j-1] )
                    temps = np.c_[temps,temp]
            
            theta = temps
    print(lr, theta)
        
        
####### stochastic gradient descent
