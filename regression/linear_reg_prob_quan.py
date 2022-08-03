import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from linear_reg_func import create_draw_network, split_create_data, init_all_theta, min_max_theta, compute_each_grad, update_each_theta_quan
# from linear_reg_func import 

########################### create network ####################################
node = 50
probp = 0.1
matrix, matrix_dict = create_draw_network(node, probp)

########################### import data #######################################
# choose a number of variables (only n>=2, n<6), and split data?
n = 5
train_test_separate = False

path = os.getcwd()
std_scaler = StandardScaler()

data1 = path + "\data\\airfoil_self_noise.dat"
col1 = ['freq', 'angle', 'chord', 'velocity', 'thickness', 'sound'] # columns of the dataset
df1 = pd.read_table(data1, sep="\t", names=col1)

if train_test_separate == True:
    train, test = train_test_split(df1, test_size=0.2)
    # choose specific columns
    if n ==2:
        X_train = train.values[:, 1::3]
        X_test = test.values[:, 1::3]
    else:       
        X_train = train.values[:, 0:n] 
        X_test = test.values[:, 0:n] 
 
    y_train = train.values[:, 5]
    y_test = test.values[:, 5]
    
    X_train = std_scaler.fit_transform(X_train)
    all_data = np.c_[X_train,y_train]

    X_test = std_scaler.fit_transform(X_test)
    all_data_test = np.c_[X_test,y_test]
else:
    # choose specific columns
    if n==2:
        X = df1.values[:, 1::3]   
    else:
        X = df1.values[:, 0:n] 
        
    y = df1.values[:, 5] 

    X = std_scaler.fit_transform(X)
    all_data = np.c_[X,y]

########################### split data to each node ###########################

datasets, max_d = split_create_data(False, all_data, node, 1)

########################### compute gradients #################################
theta_is_zero = False
visual = True
make_csv = False
err_everynode = False
seed_num = 99
epoch = 10
every_t = 1
learning_rate = [ 0.01]

col_table2 = []
errors = [] 

qs = [0.1]
# qs = [0.1,0.5,0.9]
for probq in qs:

    if len(learning_rate) > 1:
        errors = [] 

    for lr in learning_rate: 
        error = []
        
        theta0, theta = init_all_theta(node, X.shape[1], theta_is_zero)
        mn_mx = min_max_theta(theta0, theta, [])
        
        for t in range(epoch):
            all_loss = 0
            for d in range(max_d):
                all_grad0, all_grad, all_cost, all_c, ne = compute_each_grad(node-1, d, theta0, theta, datasets) 
                all_loss = all_c/ne
                
                if ne < node:
                    continue
                theta0, theta = update_each_theta_quan(lr,matrix_dict, node-1, theta0, theta, all_grad0, all_grad, mn_mx)

                mn_mx = min_max_theta(theta0, theta, mn_mx)
                if t%every_t==0:
                    if err_everynode == True:
                        error.append(all_cost)
                    else:
                        error.append(all_loss)
         
        errors.append(error)


## visualisation3
## error2
# if visual == True:
#     if len(learning_rate) == 1:
#         if err_everynode == True:
#             for i in range(0, df777.shape[1], node):
#                 converge2(df777.iloc[:,i:i+node], learning_rate[0], probp, probq, err_everynode)
#         else:
#             converge2(df777, learning_rate[0], probp, probq, err_everynode)