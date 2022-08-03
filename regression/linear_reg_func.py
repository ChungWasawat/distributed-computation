from math import floor, ceil

import numpy as np
from numpy.random import randn, randint, seed, normal, uniform, shuffle 

import networkx as nx
import matplotlib.pyplot as plt

##############################################################################
## compute function
def init_theta(seed_num: int, nrow: int) -> np.array:
    seed(seed_num)
    theta = randn(1, nrow).flatten()
    return theta[0], theta[1:]

def init_all_theta(n_node, ncol, theta_is_zero, seed_num=99):
    theta0, theta = [], []
    for _ in range(n_node):
        if theta_is_zero == True:    
            theta0.append(0)   
            theta.append(np.zeros(ncol))
        else:
            tt0, tt = init_theta(seed_num, ncol+1)
            theta0.append(tt0)
            theta.append(tt)
    return theta0, theta

def init_all_minmax_theta(n_node, nrow, theta_is_zero, seed_num=99):
    theta0, theta = [], []
    for _ in range(n_node):
        if theta_is_zero == True:    
            theta0.append(0)   
            theta.append(np.zeros(nrow))
        else:
            tt0, tt = init_theta(seed_num, nrow+1)
            theta0.append(tt0)
            theta.append(tt)
    return theta[0], theta[1:]

def y_hat(w: np.array, X: np.array, b: np.array) -> np.array:
    X_T = X.reshape(X.size,1)
    xx = np.dot(w.reshape(1, w.size), X_T).flatten() + b
    return xx

def compute_grad(y_h,y ,X_T):
    loss = y_h-y
    grad = loss * X_T
    return grad.flatten()

def cost(y_h,y):
    return 0.5 * ((y_h-y)**2)

def min_max_theta(theta0, theta, mn_mx):
    temp = []
    if mn_mx == []:  
        for t1 in range(len(theta)):
            tp = []
            #theta0 + other thetas
            for t2 in range(theta[0].size+1):
                if t2 == 0:
                    if theta0[t1] >0:
                        
                        mn,mx = theta0[t1] , theta0[t1]  
                    else:
                        mn,mx = theta0[t1], theta0[t1]
                else:
                    if theta[t1][t2-1] >0:
                        mn,mx = theta[t1][t2-1], theta[t1][t2-1]
                    else:
                        mn,mx = theta[t1][t2-1], theta[t1][t2-1]
                
                tp.append([mn,mx])
            temp.append(tp)
        return temp
    else:
        for t1 in range(len(mn_mx)):
            for t2 in range(len(mn_mx[0])):
                if t2 ==0:
                    # min
                    if mn_mx[t1][t2][0] > theta0[t1]:
                        mn_mx[t1][t2][0] = theta0[t1]
                    # max
                    elif mn_mx[t1][t2][1] < theta0[t1]:
                        mn_mx[t1][t2][1] = theta0[t1]
                else:
                    # min
                    if mn_mx[t1][t2][0] > theta[t1][t2-1]:
                        mn_mx[t1][t2][0] = theta[t1][t2-1]
                    # max
                    elif mn_mx[t1][t2][1] < theta[t1][t2-1]:
                        mn_mx[t1][t2][1] = theta[t1][t2-1]
        return mn_mx

##############################################################################
## compute gradients for each node with one theta from master
def compute_each_grad_one_theta(node:int, order:int, theta0:list, theta:list, dataset:list):
    if node ==0:
        try:   
            x = dataset[node][order][0:-1]
            y = dataset[node][order][-1]
            t0 = theta0[0]
            t= theta[0]
            
            a = y_hat(t, x, t0)
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
            all_grad0, all_grad, all_cost, all_c, n_cost = compute_each_grad_one_theta(node-1, order, theta0, theta, dataset) # ne for dataset is Not Equal
            x = dataset[node][order][0:-1]
            y = dataset[node][order][-1]
            t0 = theta0[0]
            t= theta[0]
            a = y_hat(t, x, t0)
            g = compute_grad(a, y, x)
            
            all_grad0.append((a-y)[0])
            all_grad.append(g)            
            all_cost.append( cost(a,y)[0] )           
            all_c += cost(a,y)[0]
            return all_grad0, all_grad, all_cost, all_c, n_cost+1
        except:
            all_cost.append(0)
            return all_grad0, all_grad, all_cost, all_c, n_cost
## compute gradients for each node
def compute_each_grad(node:int, order:int, theta0:list, theta:list, dataset:list):
    if node ==0:
        try:          
            x = dataset[node][order][0:-1]
            y = dataset[node][order][-1]
            t0 = theta0[node]
            t= theta[node]
            a = y_hat(t, x, t0)
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
            all_grad0, all_grad, all_cost, all_c, n_cost = compute_each_grad(node-1, order, theta0, theta, dataset) # ne for dataset is Not Equal
            x = dataset[node][order][0:-1]
            y = dataset[node][order][-1]
            t0 = theta0[node]
            t= theta[node]
            a = y_hat(t, x, t0)
            g = compute_grad(a, y, x)
            
            all_grad0.append((a-y)[0])
            all_grad.append(g)            
            all_cost.append( cost(a,y)[0] )           
            all_c += cost(a,y)[0]
            return all_grad0, all_grad, all_cost, all_c, n_cost+1
        except:
            all_cost.append(0)
            return all_grad0, all_grad, all_cost, all_c, n_cost
## update thetas for all nodes        
def update_each_theta(lr:int, network:dict, node:int, theta0:list, theta:list, all_grad0:list, all_grad:list, probq:int, success_comm:list):
    if node ==0:
        try:
            temp_theta0 = theta0[node]
            temp_theta = theta[node]
            
            # add probability of failure
            q = 0
            for i in network[node]:
                x = uniform(0,1)         
                if x <= probq:
                    temp_theta0 += theta0[i]
                    temp_theta += theta[i]
                    q+=1
            
            # find avg value of theta
            n_nodes = q + 1 # include itself
            temp_theta0 /= n_nodes
            temp_theta /= n_nodes
            
            success_comm[node] += n_nodes
            #update theta
            new_theta0 = theta0.copy()
            new_theta = theta.copy()

            new_theta0[node] = (temp_theta0 - (lr * all_grad0[node]) )
            new_theta[node] = temp_theta - (lr * all_grad[node])
            return new_theta0, new_theta, success_comm
        except:
            new_theta0=theta0.copy()
            new_theta=theta.copy()
            return new_theta0, new_theta, success_comm
    else:
        try:   
            # go to node0 first to start update
            new_theta0, new_theta, success_comm = update_each_theta(lr,network,node-1,theta0,theta,all_grad0,all_grad,probq, success_comm) 
            
            temp_theta0 = theta0[node]
            temp_theta = theta[node]
 
            q = 0
            for i in network[node]:
                x = uniform(0,1)         
                if x <= probq:
                    temp_theta0 += theta0[i]
                    temp_theta += theta[i]
                    q+=1

            n_nodes = q + 1
            temp_theta0 /= n_nodes
            temp_theta /= n_nodes
 
            success_comm[node] += n_nodes
            new_theta0[node] = (temp_theta0 - (lr * all_grad0[node]) )
            new_theta[node] = temp_theta - (lr * all_grad[node])
            return new_theta0, new_theta, success_comm
        except:
            return new_theta0, new_theta, success_comm 
## update thetas for all nodes -quantization
def update_each_theta_quan(lr, network, node, theta0, theta, all_grad0, all_grad, mn_mx):
    if node ==0:
        try:
            temp_theta0 = theta0[node]
            temp_theta = theta[node]

            q = 0
            for i in network[node]:
                for j in range(len(mn_mx[0])):
                    x = uniform(0,1)  
                    
                    if (mn_mx[i][j][1] - mn_mx[i][j][0]) ==0:
                        continue
                    
                    if j ==0:
                        min_prob = (mn_mx[i][j][1] - theta0[i] ) / (mn_mx[i][j][1] - mn_mx[i][j][0])
                        max_prob = (theta0[i] - mn_mx[i][j][0] ) / (mn_mx[i][j][1] - mn_mx[i][j][0])
                        
                        if x < min_prob:
                            temp_theta0 += mn_mx[i][j][0]
                            q+=1
                        elif x >= min_prob:
                            temp_theta0 += mn_mx[i][j][1]
                            q+=1
                    else:
                        min_prob = (mn_mx[i][j][1] - theta[i][j-1] ) / (mn_mx[i][j][1] - mn_mx[i][j][0])
                        max_prob = (theta[i][j-1] - mn_mx[i][j][0] ) / (mn_mx[i][j][1] - mn_mx[i][j][0])
                        
                        if x < min_prob:
                            temp_theta[j-1] += mn_mx[i][j][0]
                            q+=1
                        elif x >= min_prob:
                            temp_theta[j-1] += mn_mx[i][j][1]
                            q+=1                
            
            # find avg value of theta
            n_nodes = q + 1
            temp_theta0 /= n_nodes
            temp_theta /= n_nodes
  
            #update theta
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
            # go to node0 first to start update
            new_theta0, new_theta = update_each_theta_quan(lr,network,node-1,theta0,theta,all_grad0,all_grad,mn_mx) 
            
            temp_theta0 = theta0[node]
            temp_theta = theta[node]
 
            q=0
            for i in network[node]:
                for j in range(len(mn_mx[0])):
                    x = uniform(0,1)  
                    
                    if (mn_mx[i][j][1] - mn_mx[i][j][0]) ==0:
                        continue
                    
                    if j ==0:
                        min_prob = (mn_mx[i][j][1] - theta0[i] ) / (mn_mx[i][j][1] - mn_mx[i][j][0])
                        max_prob = (theta0[i] - mn_mx[i][j][0] ) / (mn_mx[i][j][1] - mn_mx[i][j][0])
                        
                        if x < min_prob:
                            temp_theta0 += mn_mx[i][j][0]
                            q+=1
                        elif x >= min_prob:
                            temp_theta0 += mn_mx[i][j][1]
                            q+=1
                    else:
                        min_prob = (mn_mx[i][j][1] - theta[i][j-1] ) / (mn_mx[i][j][1] - mn_mx[i][j][0])
                        max_prob = (theta[i][j-1] - mn_mx[i][j][0] ) / (mn_mx[i][j][1] - mn_mx[i][j][0])
                        
                        if x < min_prob:
                            temp_theta[j-1] += mn_mx[i][j][0]
                            q+=1
                        elif x >= min_prob:
                            temp_theta[j-1] += mn_mx[i][j][1]
                            q+=1     
                            
            n_nodes = q + 1
            temp_theta0 /= n_nodes
            temp_theta /= n_nodes    
 
            new_theta0[node] = (temp_theta0 - (lr * all_grad0[node]) )
            new_theta[node] = temp_theta - (lr * all_grad[node])
            return new_theta0, new_theta
        except:
            return new_theta0, new_theta  

##############################################################################
## random edges of nodes      
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
                #print(x)
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
## check all nodes are connected or not        
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
## create edges for isolated nodes
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
                        print("v =", v[0]," i =", i)
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
                        print("v =", v[0]," i =", i)
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
## create & draw network
def create_draw_network(n_node, probp):
    #create
    network_matrix, network_matrix_dict = random_network(n_node, probp)
    new_matrix, new_matrix_dict = create_path(network_matrix, network_matrix_dict)
    #draw
    temp_nx = nx.MultiGraph()
    G = nx.from_numpy_matrix(new_matrix, parallel_edges=False, create_using=temp_nx)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    nx.draw_networkx(G)
    plt.show()
    return new_matrix, new_matrix_dict

##############################################################################
## split dataset
def o_split(all_data, node):
    all_rows = all_data.shape[0]
    divided_n = floor(all_rows/node)
    max_divided_n = ceil(all_rows/node) 
    remain_d = all_rows%node

    datasets = []
    start, stop = 0, divided_n

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
        
    return datasets, max_divided_n

## split dataset & create fake data
def split_create_data(fake:bool, all_data, node:int, new_node:int, sample:int = 1):
    if fake == True:
        all_rows = all_data.shape[0]
        divided_n = floor(all_rows/node)
        max_divided_n = ceil(all_rows/node) 
        remain_d = all_rows%node

        datasets = []
        start, stop = 0, divided_n
        # divide the entire dataset to n nodes
        for n in range(node):   
            # shuffle for sgd
            seed(99)
            shuffle(all_data[ start : stop, : ])
            datasets.append( all_data[ start : stop, : ] )
            start, stop = stop, stop+divided_n  
            if n ==0:
                remain_data = all_data[ start : start+sample, : ]
            else:
                remain_data = np.concatenate((remain_data, all_data[ start : start+sample, : ]))
        
        # add remained data to the pool
        if stop != all_rows:
            remain_data = np.concatenate((remain_data, all_data[start: , :]))
        
        # create new data from normal distribution of the pool
        for nn in range(new_node):
            for n in range(remain_data.shape[1]):
                mean = np.mean(remain_data[:,n])
                std = np.std(remain_data[:,n])
                
                rand_data = normal(mean, std, divided_n).reshape(divided_n,1)
                if n == 0:
                    temp_data = rand_data
                else:
                    temp_data = np.concatenate((temp_data, rand_data), axis=1)
                    
            x = randint(0,node)    
            datasets[x] = temp_data
        return datasets, divided_n
    
    else:
        all_rows = all_data.shape[0]
        divided_n = floor(all_rows/node)
        max_divided_n = ceil(all_rows/node) 
        remain_d = all_rows%node
    
        datasets = []
        start, stop = 0, divided_n
    
        datasets = []
        start, stop = 0, divided_n
        # divide the entire dataset to n nodes
        for n in range(node):   
            if remain_d > 0:
                stop+=1
                remain_d -= 1
            # shuffle for sgd
            seed(99)
            shuffle(all_data[ start : stop, : ])
            datasets.append( all_data[ start : stop, : ] )
            start, stop = stop, stop+divided_n
            
        return datasets, max_divided_n
    
##############################################################################
## visualisation
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
   