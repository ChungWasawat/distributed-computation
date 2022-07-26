import os

from math import floor, ceil

import numpy as np
import pandas as pd
from numpy.random import seed
from numpy.random import randn
from numpy.random import uniform
from numpy.random import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import networkx as nx
import matplotlib.pyplot as plt

def theta_init(seed_num: int, row: int) -> np.array:
    # seed(seed_num)
    theta = randn(1, row).flatten()
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

def compute_all_grad(node:int, order:int, theta0:list, theta:list, dataset:list):
    if node ==0:
        try:          
            x = dataset[node][order][0:-1]
            y = dataset[node][order][-1]
            t0 = theta0[node]
            t= theta[node]
            a = sigmoid(t, x, t0)
            g = compute_grad(a, y, x)
            
            all_grad0 = []
            all_grad = []
            all_cost = []
         
            all_grad0.append((a-y)[0])
            all_grad.append(g)           
            all_cost.append( cost(a,y)[0] )
            all_c = cost(a,y)[0]
            return all_grad0, all_grad, all_cost, all_c, 1
        except:
            all_grad0 = []
            all_grad = []
            all_cost = []
            
            all_grad0.append(0)
            all_grad.append(np.zeros(theta[node].size))           
            all_cost.append(0)            
            return all_grad0, all_grad, all_cost, 0, 0
    else:
        try:
            all_grad0, all_grad, all_cost, all_c, n_cost = compute_all_grad(node-1, order, theta0, theta, dataset) # ne for dataset is Not Equal
            x = dataset[node][order][0:-1]
            y = dataset[node][order][-1]
            t0 = theta0[node]
            t= theta[node]
            a = sigmoid(t, x, t0)
            g = compute_grad(a, y, x)
            
            all_grad0.append((a-y)[0])
            all_grad.append(g)            
            all_cost.append( cost(a,y)[0] )           
            all_c += cost(a,y)[0]
            return all_grad0, all_grad, all_cost, all_c, n_cost+1
        except:
            all_cost.append(0)  
            return all_grad0, all_grad, all_cost, all_c, n_cost

def update_all_theta(lr:int, network:dict, node:int, theta0:list, theta:list, all_grad0:list, all_grad:list):
    if node ==0:
        try:
            temp_theta0 = theta0[node]
            temp_theta = theta[node]
            
            for i in network[node]:
                temp_theta0 += theta0[i]
                temp_theta += theta[i]
            
            n_nodes = len(network[node]) + 1
            temp_theta0 /= n_nodes
            temp_theta /= n_nodes

            new_theta0 = theta0.copy()
            new_theta = theta.copy()
            
            new_theta0[node] = (temp_theta0 - (lr * all_grad0[node]) )
            new_theta[node] = temp_theta - (lr * all_grad[node])
            return new_theta0, new_theta
        except:
            new_theta0=theta0.copy()
            new_theta=theta.copy()
            return new_theta0, new_theta
    else:
        try:           
            new_theta0, new_theta = update_all_theta(lr,network,node-1,theta0,theta,all_grad0,all_grad) 
            
            temp_theta0 = theta0[node]
            temp_theta = theta[node]
 
            for i in network[node]:
                temp_theta0 += theta0[i]
                temp_theta += theta[i]

            n_nodes = len(network[node]) + 1
            temp_theta0 /= n_nodes
            temp_theta /= n_nodes

            new_theta0[node] = (temp_theta0 - (lr * all_grad0[node]) )
            new_theta[node] = temp_theta - (lr * all_grad[node])
            return new_theta0, new_theta
        except:
            return new_theta0, new_theta 

def converge(error, step, lr, prob, node="all nodes"):
    plt.figure()
    plt.xlim(0, step)
    plt.plot(error, color = 'b')
    if type(node) == "int":
        title = f"learning rate= {lr}, probability= {prob}, at node{node}"
    else:
        title = f"learning rate= {lr}, probability= {prob}, {node}"
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("error")
    plt.show()
    
def converge2(df_error, lr, probp, probq, all_node):  
    df_error.plot(grid=True) 
    if all_node == True:
        title = f"learning rate= {lr}, probability p= {probp}, probability q= {probq}"
        
    else:
        title = f"learning rate= {lr}, probability p= {probp}"
        
    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("error")
    plt.show()
    
def random_network(node: int, p: float) -> np.array:
    nw = np.zeros((node,node))

    route = {}
    for i in range(node):
        route[i] = []

    c = (node **2)
    for i in range(node-1,0,-1):
        c -= i

    i,j = 0,0
    while c > 0:
        if i==j:
            nw[i,j] = 1
            nw[j,i] = 1
        else:
            x = uniform(0,1)
            if x <= p:
                nw[i,j] = 1
                nw[j,i] = 1 
                if j not in route[i]:
                    route[i].append(j)  
                    route[j].append(i)
        j+=1
        if j == node:
            i += 1
            j = i
        c -=1
    return nw, route

def visit_adj(adj_m:dict, node:int, neighbour:int, visited:list) -> list:
    if adj_m[node] == []:
        visited.append(node)
        return visited
    elif adj_m[node] == [] and node == len(adj_m.keys())-1:
        return visited 
    if node not in visited:
        visited.append(node)
    if adj_m[node][neighbour] in visited and adj_m[node][neighbour] == adj_m[node][-1]:
        return visited
    elif adj_m[node][neighbour] in visited :
        return visit_adj(adj_m, node, neighbour+1, visited) 
    elif adj_m[node][neighbour] not in visited and adj_m[node][neighbour] == adj_m[node][-1]:
        return visit_adj(adj_m, adj_m[node][neighbour], 0, visited)
    else:
        visited = visit_adj(adj_m, adj_m[node][neighbour], 0, visited) 
        return visit_adj(adj_m, node, neighbour+1, visited)

def create_path(adj_m: np.array, adj_m_d:dict):
    temp_v = []
    
    for i in range(len(adj_m_d.keys())):
        v = visit_adj(adj_m_d, i, 0, [])  
        if len(v) == len(adj_m_d.keys()):
            break                 
        if len(v) > 1:
            if i == len(adj_m_d.keys())-1:
                if i-1 not in v:
                    adj_m_d[i].append(i-1)
                    adj_m_d[i-1].append(i)
                    adj_m[i,i-1] = 1
                    adj_m[i-1,i] = 1
                else:
                    if temp_v == []:
                        temp_v = v.copy()
                    elif len( set(temp_v).intersection(set(v)) ) == 0:
                        adj_m_d[v[0]].append(temp_v[0])
                        adj_m_d[temp_v[0]].append(v[0])
                        adj_m[v[0],temp_v[0]] = 1
                        adj_m[temp_v[0],v[0]] = 1     
                        temp_v = []                   
                    else:
                        continue
            else:
                if i+1 not in v:
                    adj_m_d[i].append(i+1)
                    adj_m_d[i+1].append(i)
                    adj_m[i,i+1] = 1
                    adj_m[i+1,i] = 1                   
                else:
                    if temp_v == []:
                        temp_v = v.copy()
                    elif len( set(temp_v).intersection(set(v)) ) == 0:
                        adj_m_d[v[0]].append(temp_v[0])
                        adj_m_d[temp_v[0]].append(v[0])
                        adj_m[v[0],temp_v[0]] = 1
                        adj_m[temp_v[0],v[0]] = 1     
                        temp_v = []                   
                    else:
                        continue
        else:
            if i == len(adj_m_d.keys())-1:
                adj_m_d[i].append(i-1)
                adj_m_d[i-1].append(i)
                adj_m[i,i-1] = 1
                adj_m[i-1,i] = 1
            else:
                adj_m_d[i].append(i+1)
                adj_m_d[i+1].append(i)
                adj_m[i,i+1] = 1
                adj_m[i+1,i] = 1

    return adj_m, adj_m_d

#################################################################
## import data
path = os.getcwd()

data1 = path + "\data\\data_banknote_authentication.txt"
col1 = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

df1 = pd.read_table(data1, sep=",", names=col1)
std_scaler = StandardScaler()

# choose a number of variables !only n>=2, n<5
n = 4

if n==2:
    X = df1.values[:, 1:3]  
else:
    X = df1.values[:, 0:n] 

Y = df1.values[:, 4]

X = std_scaler.fit_transform(X)
all_data = np.c_[X,Y]

#################################################################
## crete network
node = 50
probp = 0.5
network_matrix, network_matrix_dict = random_network(node, probp)
new_matrix, new_matrix_dict = create_path(network_matrix, network_matrix_dict)

## visualisation
## graph
temp_nx = nx.MultiGraph()
G = nx.from_numpy_matrix(new_matrix, parallel_edges=False, create_using=temp_nx)
G.remove_edges_from(list(nx.selfloop_edges(G)))
nx.draw_networkx(G)
plt.show()

#################################################################
## logistic regression
model = LogisticRegression()
model.fit(X, Y)

print("from logistic regression", model.intercept_.round(decimals=3), model.coef_.round(decimals=3))


##########################################################
## sgd
theta_is_zero = False
visual = True
make_csv = False
err_everynode = False
seed_num = 99
epoch = 15
every_t = 1
learning_rate = [ 0.01]
# learning_rate = [0.005, 0.01, 0.05, 0.1]

nrows = all_data.shape[0]
divided_n = floor(nrows/node)
max_divided_n = ceil(nrows/node) 
remain_d = nrows%node

datasets = []
start, stop = 0, divided_n
# divide the entire dataset to n nodes
for n in range(node):   
    if remain_d > 0:
        stop+=1
        remain_d -= 1
    # shuffle for sgd
    # seed(seed_num)
    shuffle(all_data[ start : stop, : ])
    datasets.append( all_data[ start : stop, : ] )
    start, stop = stop, stop+divided_n

col_table2 = []
errors = [] 
df777 = pd.DataFrame()
qs = [0.1,0.5,0.9]

for probq in qs:
    if len(learning_rate) > 1:
        errors = [] 
        
    for lr in learning_rate: 
        theta0, theta = [], []
        error = []
        for n in range(node):
            if theta_is_zero == True:    
                theta0.append(0)   
                theta.append(np.zeros(X.shape[1]))
            else:
                tt0, tt = theta_init(seed_num, X.shape[1]+1)
                theta0.append(tt0)
                theta.append(tt)
        
        for t in range(epoch):
            all_loss = 0
            for d in range(max_divided_n):
                all_grad0, all_grad, all_cost, all_c, ne = compute_all_grad(node-1, d, theta0, theta, datasets) 
                all_loss = all_c/ne
                
                if ne < node:
                    continue            
                theta0, theta = update_all_theta(lr,new_matrix_dict, node-1, theta0, theta, all_grad0, all_grad)
                
                if t%every_t==0:
                    if err_everynode == True:
                        error.append(all_cost)
                    else:
                        error.append(all_loss)
                        
        str_probp = str(probp)[0]+str(probp)[2]
        str_probq = str(probq)[0]+str(probq)[2]
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


        errors.append(error)

    ## visualisation2
    ## error
    if visual == True:
        if len(learning_rate) > 1:
            if err_everynode == True:
                for lr in range(len(learning_rate)):
                    all_e = np.array(errors[lr])
                    for n in range(node):
                        converge(all_e[:,n], all_e.shape[0], learning_rate[lr], probp, n)
            else:
                for lr in range(len(learning_rate)):
                    all_e = np.array(errors[lr])
                    converge(all_e, len(errors[lr]), learning_rate[lr], probp)

## visualisation3
## error2
if visual == True:
    if len(learning_rate) == 1:
        if err_everynode == True:
            for i in range(0, df777.shape[1], node):
                converge2(df777.iloc[:,i:i+node], learning_rate[0], probp, probq, err_everynode)
        else:
            converge2(df777, learning_rate[0], probp, probq, err_everynode)
        
##########################################################
## csv
if make_csv == True:
    data2 = path + "\csv\\"
    if len(learning_rate) > 1:

        if err_everynode == False:
            col_table2 = ["loss"]
            df9999 = pd.DataFrame(errors[0], columns=col_table2) #lr = 0.01
            str_probp = str(probp)[0]+str(probp)[2]
            df9999.to_csv(data2+f"cla_decen_sgd_node{node}_prob{str_probp}.csv", index=False)
        else:
            str_probp = str(probp)[0]+str(probp)[2]
            col_table2 = [f"node{z} p{str_probp}" for z in range(node)]
            df9999 = pd.DataFrame(errors[0], columns=col_table2) #0=lr 0.01 1=lr 0.1
            df9999.to_csv(data2+f"cla_decen_sgd_node{node}_prob{str_probp}_each_err.csv", index=False)
    else:
        if err_everynode == False:
            df777.to_csv(data2+f"cla_decen_sgd_n{node}_p{str_probp}_allq.csv", index=False)
        else:
            df777.to_csv(data2+f"cla_decen_sgd_n{node}_p{str_probp}_allq_each_err.csv", index=False)
        


