import numpy as np
from utils import *

#Stochastic Gradient Descent Algorithm
#INPUTS:
#a0: intial guess for parameters a (can be randomly assigned or zero vector)
#C: coordinate data [x,y] x grouped and y grouped
#K: size of stochastic subset of data to gradient-step in
#eta: learning-rate/stepping size
#num: number of iterations/steps
#OUTPUT:

def SGDdb(a0, C, eta, K, num, p):
    #size of Operator
    m = np.size(C,1)
    n = np.size(a0)
    
    X = F(C[0]) #NEW
    
    A = np.zeros((num,n))
    A[0] = a0
    L = [i for i in range(m)]
    
    #initialize subset choice
    L_k = rand_subset(L,K)
    
    for l in range(1,num):
        #updating new 'a' value
        G = np.zeros(n)
        
        #initialize subset choice
        L_k = rand_subset(L,K)
        
        for j in L_k:
            #store gradient contribution
            g = grad_phi(A[l-1],X[:,j],C[1][j]) #MEW
            #add to overall gradient
            G = G + g
        
        G = G/K
        A[l] = A[l-1] - eta*G/(l**p)

    print('Gradient:',G)
    return A