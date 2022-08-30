import os

import numpy as np
from numpy.random import shuffle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from linear_reg_func import create_draw_network, split_create_data, init_all_theta, min_max_theta
from linear_reg_func import update_each_theta, update_each_theta_quan, compute_each_grad, converge2

# edit update quan
########################### create network ####################################
node = 5
probp = 0.5
matrix, matrix_dict = create_draw_network(node, probp)

########################### import data #######################################
# choose a number of variables (only n>=2, n<6), and split data?
col = 2
train_test_separate = False

path = os.getcwd()
std_scaler = StandardScaler()

data1 = path + "\data\\airfoil_self_noise.dat"
col1 = ['freq', 'angle', 'chord', 'velocity', 'thickness', 'sound'] # columns of the dataset
df1 = pd.read_table(data1, sep="\t", names=col1)

if train_test_separate == True:
    train, test = train_test_split(df1, test_size=0.2)
    # choose specific columns # angle+thickness are high correlation
    if col ==2:
        X_train = train.values[:, 1::3]
        X_test = test.values[:, 1::3]
    else:       
        X_train = train.values[:, 0:col] 
        X_test = test.values[:, 0:col] 
 
    y_train = train.values[:, -1]
    y_test = test.values[:, -1]
    
    # X_train = std_scaler.fit_transform(X_train)
    all_data = np.c_[X_train,y_train]

    # X_test = std_scaler.fit_transform(X_test)
    all_data_test = np.c_[X_test,y_test]
else:
    # choose specific columns
    if col==2:
        X = df1.values[:, 1::3]   
    else:
        X = df1.values[:, 0:col] 
        
    y = df1.values[:, -1] 

    # X = std_scaler.fit_transform(X)
    all_data = np.c_[X,y]

########################### split data to each node ###########################
# 1st boolean for create fake data
# all data = entire training data
# node = a number of nodes
mu = np.mean(all_data, axis=0)
sigma = np.std(all_data, axis=0)
lim=[1.5,1.7]
fake = False
new_fake_node = 1
datasets, max_d = split_create_data(fake, all_data, node, new_fake_node, mu, sigma, lim)

########################### compute gradients #################################
# default 
col_table2 = []
errors = [] 
df777 = pd.DataFrame()

#changable
theta_is_zero = False
visual = True
make_csv = False
err_everynode = False
seed_num = 99
epoch = 2
every_t = 1
learning_rate = [ 0.01]
mn_mx = []

qs = [0.9]
## decentralised sgd
for probq in qs:

    if len(learning_rate) > 1:
        errors = [] 

    for lr in learning_rate: 
        error = []
        #initial theta
        theta0, theta = init_all_theta(node, X.shape[1], theta_is_zero)
        succ_comm = [0]*(node)

        mn_mx = min_max_theta(theta0, theta, [])
        for t in range(epoch):
            all_loss = 0
            for d in range(max_d):
                #compute gradient
                all_grad0, all_grad, all_cost, all_c, ne = compute_each_grad(node-1, d, theta0, theta, datasets) 
                all_loss = all_c/ne
                # skip this iteration because data of all nodes is not equal
                if ne < node:
                    continue
                #update theta
                theta0, theta, succ_comm = update_each_theta(lr,matrix_dict, node-1, theta0, theta, all_grad0, all_grad, probq, succ_comm)
                #update min max
                mn_mx = min_max_theta(theta0, theta, mn_mx)
                
                if t%every_t==0:
                    if err_everynode == True:
                        error.append(all_cost)
                    else:
                        error.append(all_loss)
                
        errors.append(error)
        print(f"avg successful communication for decen sgd with n={node}, q={probq}",np.mean(succ_comm))
        print(f"n={node}, avg of theta ", round(np.mean(theta0),3), np.round(np.mean(theta, axis=0),3) )                  
        ## visualisation   
        str_probp = str(probp)[0]+str(probp)[2]
        if probq !=1:
            str_probq = str(probq)[0]+str(probq)[2]
        else:
            str_probq = str(probq)
        if err_everynode == True:
            for n in range(node):
                col_table2.append(f"loss_n{n}_p{str_probp}_q{str_probq}")
            if df777.size ==0:
                df777 = pd.DataFrame(error, columns=col_table2)
                col_table2 = []
            else:
                temp_df777 = pd.DataFrame(error, columns=col_table2)
                df777 = pd.concat([df777,temp_df777 ], axis=1)
                col_table2 = []
        else:         
            col_table2.append(f"loss_p{str_probp}_q{str_probq}")
            if df777.size ==0:
                df777 = pd.DataFrame(error, columns=col_table2)
                col_table2 = []
            else:
                temp_df777 = pd.DataFrame(error, columns=col_table2)
                df777 = pd.concat([df777,temp_df777 ], axis=1)
                col_table2 = [] 

## plot
if visual == True:
    if len(learning_rate) == 1:
        if err_everynode == True:
            for i in range(0, df777.shape[1], node):
                converge2(df777.iloc[:,i:i+node], learning_rate[0], probp, probq, err_everynode)
        else:
            converge2(df777, learning_rate[0], probp, probq, err_everynode)  

# default 
col_table2 = []
errors = [] 
df777 = pd.DataFrame()

# qs = [0.1,0.5,0.9]
# learning_rate = [0.1]
div = 30
epoch = 5
# max_d = 3

# shuffle data to do stochastic gradient descent
for d in datasets:
    shuffle(d)

# quantization
for probq in qs:

    if len(learning_rate) > 1:
        errors = [] 

    for lr in learning_rate: 
        error = []
        
        theta0, theta = init_all_theta(node, X.shape[1], theta_is_zero) 
        for t in range(epoch):
            all_loss = 0
            for d in range(max_d):
                all_grad0, all_grad, all_cost, all_c, ne = compute_each_grad(node-1, d, theta0, theta, datasets) 
                all_loss = all_c/ne
                
                if ne < node:
                    continue
                theta0, theta = update_each_theta_quan(lr,matrix_dict, node-1, theta0, theta, all_grad0, all_grad, mn_mx, div)
                if theta0 == 0 and theta == 0:
                    break
                
                if d == 0 or d== 1 or d==max_d-2:
                    print(t,all_loss)
                if t%every_t==0:
                    if err_everynode == True:
                        error.append(all_cost)
                    else:
                        error.append(all_loss)
            if theta0 == 0 and theta == 0:
                break
        errors.append(error)
        
        ## visualisation   
        str_probp = str(probp)[0]+str(probp)[2]
        if probq !=1:
            str_probq = str(probq)[0]+str(probq)[2]
        else:
            str_probq = str(probq)
        if err_everynode == True:
            for n in range(node):
                col_table2.append(f"loss_n{n}_p{str_probp}_q{str_probq}")
            if df777.size ==0:
                df777 = pd.DataFrame(error, columns=col_table2)
                col_table2 = []
            else:
                temp_df777 = pd.DataFrame(error, columns=col_table2)
                df777 = pd.concat([df777,temp_df777 ], axis=1)
                col_table2 = []
        else:         
            col_table2.append(f"loss_p{str_probp}_q{str_probq}")
            if df777.size ==0:
                df777 = pd.DataFrame(error, columns=col_table2)
                col_table2 = []
            else:
                temp_df777 = pd.DataFrame(error, columns=col_table2)
                df777 = pd.concat([df777,temp_df777 ], axis=1)
                col_table2 = [] 
  
## plot
if visual == True:
    if len(learning_rate) == 1:
        if err_everynode == True:
            for i in range(0, df777.shape[1], node):
                converge2(df777.iloc[:,i:i+node], learning_rate[0], probp, probq, err_everynode)
        else:
            converge2(df777, learning_rate[0], probp, probq, err_everynode)  
            