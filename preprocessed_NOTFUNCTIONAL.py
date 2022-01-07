import numpy as np
from utils import *


#Main Evolutionary Stochastic Gradient Descent Algorithm
#INPUTS:
#a0: intial guess for parameters a (can be randomly assigned or zero vector)
#C: coordinate data [x,y] x grouped and y grouped
#K: size of stochastic subset of data to gradient-step in
#eta: learning-rate/stepping size
#num: number of iterations/steps
#T : number of top (data) members to keep before randomizing next subset via SGD
#OUTPUT:
#A: Matrix of rows of a values throughout iteration

def evSGDs(a0, C, eta, K, num, p, T):
    #size of Operator
    m = np.size(C,1)
    n = np.size(a0)
    
    X = F(C[0]) #NEW
    X, M, v_s = preprocess(X,n)
    
    A = np.zeros((num,n))
    A[0] = a0
    L = [i for i in range(m)]
    
    #initialize subset choice
    L_k = rand_subset(L,K)
    
    for l in range(1,num):
        #updating new 'a' value
        G = np.zeros(n)
        g_size = np.zeros(K)
        i = 0
        
        #initialize subset choice
        L_k = rand_subset(L,K)
        
        for j in L_k:
            #store gradient contribution
            g = grad_phi(A[l-1],X[:,j],C[1][j]) #NEW
            #add to overall gradient
            G = G + g
            
            #restore gradient as its magnitude (L2 Norm <- choice) for evolution decision later 
            g_size[i] = np.linalg.norm(g)
            i = i + 1

        G = G/K
        A[l] = A[l-1] - eta*G/(l**p)

        #reseting the population before next iteration
        select = selection(L_k,g_size, T)
        L_k = repopulate(select,L, K, T)
    print('Gradient:',G)
    for i in range(1,num):
        A[i] = postprocess(A[i],M,v_s,n)
    return A