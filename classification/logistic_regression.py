import os

import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randn
from numpy.random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

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
nrows= all_data.shape[0]

old_theta = []
errors = []

seed(seed_num)
shuffle(all_data)
train_x = all_data[:, :-1]
train_y = all_data[:, -1]  

for lr in learning_rate:   
    if theta_is_zero == True:
        theta0,theta = 0, np.zeros(X.shape[1])
    else:
        theta0, theta = theta_init(seed_num)
    
    old_th = []
    old_th.append(theta)
    error = []    
    
    for t in range(epoch):
        c=0
        for m in range(nrows):
            x = train_x[m]
            y = train_y[m]
            a = sigmoid(theta, x, theta0)
            g = compute_grad(a, y, x)

            theta0 = theta0 - (lr * (a-y))
            theta = theta - (lr * g)
            
            c += cost(a,y)
            
            if m%batch_size==0 :
                old_th.append(theta)   
        if t%every_t==0:
            error.append(c/nrows)
            
    print("learning rate=", lr, theta0.round(decimals=3), theta.round(decimals=3))
    temp_table = ["sgd", lr, 0, epoch, theta0]
    for z in theta:
        temp_table.append(z.round(decimals=3))
    table.append(temp_table)
        
    old_theta.append(old_th)    
    errors.append(error)
   
# visualisation
## gradient
for lr in range(len(learning_rate)):
    all_w = np.array(old_theta[lr])
    if all_w.shape[1] > 2:
        break
    contour(model.coef_.flatten(), all_w, learning_rate[lr])       
## error
for lr in range(len(learning_rate)):
    all_e = np.array(errors[lr])
    converge(all_e, len(errors[lr]), learning_rate[lr])
    
##########################################################
## sgd with mini batch
old_theta = []
errors = []

for lr in learning_rate:   
    if theta_is_zero == True:
        theta0,theta = 0, np.zeros(X.shape[1])
    else:
        theta0, theta = theta_init(seed_num)
    
    sum_theta0 = np.zeros(1)
    sum_theta = np.zeros(theta.shape)
    
    old_th = []
    old_th.append(theta)
    error = []    

    for t in range(epoch):
        for m in range(nrows):
            x = train_x[m]
            y = train_y[m]
            a = sigmoid(theta, x, theta0)
            g = compute_grad(a, y, x)
            
            sum_theta0 += (a-y)
            sum_theta += g
            
            #update depends on batch size
            if m%batch_size==0 or m==nrows-1:
                if m == nrows-1:
                    sum_theta0 /= (nrows%batch_size)
                    sum_theta /= (nrows%batch_size)
                else:
                    sum_theta0 /= batch_size  
                    sum_theta /= batch_size               
            
                theta0 = theta0 - (lr * sum_theta0)
                theta = theta - (lr * sum_theta)
            
                sum_theta0 = np.zeros(1)
                sum_theta = np.zeros(theta.shape)
                
        if t % every_t ==0:
            old_th.append(theta)              
            c = cost(a,y)
            error.append(c)
            
    print("learning rate=", lr, theta0.round(decimals=3), theta.round(decimals=3))
    temp_table = ["sgd with mini-batch", lr, batch_size, epoch, theta0]
    for z in theta:
        temp_table.append(z.round(decimals=3))
    table.append(temp_table)   
    
    old_theta.append(old_th)    
    errors.append(error)
   
# visualisation
## gradient
for lr in range(len(learning_rate)):
    all_w = np.array(old_theta[lr])
    if all_w.shape[1] > 2:
        break
    contour(model.coef_.flatten(), all_w, learning_rate[lr])  
## error
for lr in range(len(learning_rate)):
    all_e = np.array(errors[lr])
    converge(all_e, len(errors[lr]), learning_rate[lr])
    
##########################################################
## csv
data2 = path + "\csv\\cla_normal_sgd.csv"
df999 = pd.DataFrame(table, columns=col_table)
df999.to_csv(data2, index=False)






