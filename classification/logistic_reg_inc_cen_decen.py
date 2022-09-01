import os

import numpy as np
from numpy.random import uniform, shuffle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from logistic_reg_func import create_draw_network, split_create_data, converge2
from logistic_reg_func import init_all_theta, compute_each_grad_one_theta, compute_each_grad, update_each_theta

########################### create network ####################################
## for decentralised sgd
node = 15
probp = 0.9                         # probability for a number of links that connect nodes
matrix, matrix_dict = create_draw_network(node, probp)

########################### import data #######################################
col = 4                             # choose a number of variables (only n>=2, n<5)
train_test_separate = False         # separate train/test for early stopping?

path = os.getcwd()
std_scaler = StandardScaler()

data1 = path + "\data\\data_banknote_authentication.txt"
col1 = ['variance', 'skewness', 'curtosis', 'entropy', 'class'] # columns of the dataset
df1 = pd.read_table(data1, sep=",", names=col1)

if train_test_separate == True:
    train, test = train_test_split(df1, test_size=0.2)
    # choose specific columns
    if col ==2:
        X_train = train.values[:, 1::3]
        X_test = test.values[:, 1::3]
    else:       
        X_train = train.values[:, 0:col] 
        X_test = test.values[:, 0:col] 
 
    y_train = train.values[:, -1]
    y_test = test.values[:, -1]
    
    X_train = std_scaler.fit_transform(X_train)
    all_data = np.c_[X_train,y_train]

    X_test = std_scaler.fit_transform(X_test)
    all_data_test = np.c_[X_test,y_test]
else:
    # choose specific columns
    if col==2:
        X = df1.values[:, 1::3]   
    else:
        X = df1.values[:, 0:col] 
        
    y = df1.values[:, -1] 

    X = std_scaler.fit_transform(X)
    all_data = np.c_[X,y]

########################### split data to each node ###########################
# 1st boolean for create fake data
# all data = entire training data
# node = a number of nodes
mu = np.mean(all_data, axis=0)
sigma = np.std(all_data, axis=0)
lim=[0.05,0.15]                              #interval on normal dist of fake data
fake = False
new_fake_node = 10
datasets, max_d = split_create_data(fake, all_data, node, new_fake_node, mu, sigma, lim)

########################### compute gradients #################################
# default 
col_table2 = []
errors = [] 
df777 = pd.DataFrame()

# changable 
theta_is_zero = False       # initial theta with zero or random value
visual = True               # show graph
make_csv = False            # create csv for this data (not availble)
err_everynode = False       # separate error of each node
seed_num = 42               # seed for random everything on the model
epoch = 30                   # iteration for training model
every_t = 1                 # store error at data that % t ==0 
learning_rate = [ 0.05]     # only one for now

# probability for failure communication
qs = [0.1, 1] # 10% and no prob fail 

# threshold to stop training
conv = 0.001

# 1 for centralised sgd, 2 for decentralised sgd
case = 2
if case==1:
    ## centralised sgd
    ## only use with qs = [1] because all nodes need to finish their task before going to the next time step
    for probq in qs:
        for lr in learning_rate: 
            error = []
            #initial theta
            theta0, theta = init_all_theta(node, X.shape[1], theta_is_zero) 

            succ_comm = [0]*(node)
            cont = 0
            for t in range(epoch):
                total = 0
                for d in range(max_d):
                    #compute gradient
                    all_grad0, all_grad, all_cost, all_c, ne = compute_each_grad_one_theta(node-1, d, theta0, theta, datasets) 
                    all_loss = all_c/ne   
                    
                    # skip this iteration because data of all nodes is not equal
                    if ne < node:
                        cont = 1
                        continue
                    
                    #find average theta and update it
                    sum_grad0 = 0
                    sum_grad = np.zeros(X.shape[1])
                    nee = 0
                    for i in range(ne):
                        x = uniform(0,1) 
                        if x<=probq:
                            sum_grad0 += all_grad0[i]
                            sum_grad += all_grad[i]
                            succ_comm[i] +=1
                            nee+=1
                    
                    total += all_loss
                    total /= d+1 
                    
                    if total < conv:
                        print("before max ", "epoch= ",t," row= ",d)
                        break 
                    if nee >0:
                        avg_grad0 = sum_grad0 / nee
                        avg_grad = sum_grad / nee
                        
                        theta0[0] = theta0[0] - (lr * avg_grad0)
                        theta[0] = theta[0] - (lr * avg_grad)
                    
                    #store error
                    if t%every_t==0:
                        # separate cost of each node
                        if err_everynode == True:
                            error.append(all_cost)
                        # sum cost
                        else:
                            error.append(all_loss)
                
                #threshold for stop
                if total < conv and cont ==0:           
                    break         
                
                for ds in datasets:
                    shuffle(ds)
                 
                cont = 0
            if t == epoch-1:
                print("at max ", "epoch= ",t," row= ",d)    
                
            errors.append(error)
            print(f"avg successful communication for decen sgd with n={node}",np.mean(succ_comm))
            print(f"n={node}, avg of theta ", round(np.mean(theta0),3), np.round(np.mean(theta, axis=0),3) )
            ## visualisation   
            str_probp = str(1)
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
                    converge2(df777.iloc[:,i:i+node], learning_rate[0], 1, probq, err_everynode)
            else:
                converge2(df777, learning_rate[0], 1, probq, err_everynode)   
                   
elif case==2:
    ## decentralised sgd
    for probq in qs:
    
        if len(learning_rate) > 1:
            errors = [] 
    
        for lr in learning_rate: 
            error = []
            #initial theta
            theta0, theta = init_all_theta(node, X.shape[1], theta_is_zero)
            succ_comm = [0]*(node)
            cont = 0
            for t in range(epoch):
                total = 0
                for d in range(max_d):
                    #compute gradient
                    all_grad0, all_grad, all_cost, all_c, ne = compute_each_grad(node-1, d, theta0, theta, datasets) 
                    all_loss = all_c/ne           
                    # skip this iteration because data of all nodes is not equal
                    if ne < node:
                        cont = 1
                        continue
                    
                    total += all_loss
                    total /= d+1 
                    
                    if total < conv:
                        print("before max ", "epoch= ",t," row= ",d)
                        break 
                    
                    #update theta
                    theta0, theta, succ_comm = update_each_theta(lr,matrix_dict, node-1, theta0, theta, all_grad0, all_grad, probq, succ_comm)
    
                    if t%every_t==0:
                        if err_everynode == True:
                            error.append(all_cost)
                        else:
                            error.append(all_loss)
                #threshold for stop
                if total < conv and cont ==0:           
                    break         
                
                for ds in datasets:
                    shuffle(ds)
                 
                cont = 0
            if t == epoch-1:
                print("at max ", "epoch= ",t," row= ",d)                             
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
        