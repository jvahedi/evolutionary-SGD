import numpy as np
from utils import  *
from EA_utils import *

#Semi-Stochastic Gradient Descent Algorithm
#INPUTS:
#a0: intial guess for parameters a (can be randomly assigned or zero vector)
#C: coordinate data [x,y] x grouped and y grouped
#K: size of stochastic subset of data to gradient-step in
#eta: learning-rate/stepping size
#num: number of iterations/steps
#T : number of top (data) members to keep before randomizing next subset via SGD
#OUTPUT:
#A: Matrix of rows of a values throughout iteration

def SSGDdb(a0, C, eta, K, num, p, T):
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
    return A

# Comprehensive Evolutionary Stochastic Gradient Descent

#INPUTS:
#a0: initial guess for parameters a (can be randomly assigned or zero vector)
#C: coordinate data [x,y] x grouped and y grouped
#K: size of stochastic subset of data to gradient-step in
#eta: learning-rate/stepping size
#num: number of iterations/steps (need to account for size of population for fair comparison)
#pop_size: number of members for the evolving population
#recombination_split: split of genetic material to be acquired from either parent (e.g. 0.7 means 70% of genes from parent 1, 30% from parent 2)
#p: eta decay parameter

#OUTPUT:
#A: Matrix of best a values throughout iteration
#best_a: a value with least error


def CEvSGDdb(a0, C, eta, K, num, pop_size, recombination_split, p):
    # size of Operator
    m = np.size(C, 1)
    n = np.size(a0)


    X = F(C[0]) #NEW

    # record best parameter set (in memory for easy access)
    A = np.ndarray((num, n))

    # record params for a member until it dies (only keep for current step and immediate past step)
    current_a_vals = []
    past_a_vals = []
    best_a = []
    min_error = 10000000

    for a_i in range(pop_size):
        current_a_vals.append(a0.copy())

    # indices
    L = [i for i in range(m)]

    # initialize members of population (subset choices) & age list (all start at zero)
    L_k_list = []
    age_list = []
    for L_k_i in range(pop_size):
        L_k_list.append(np.sort(rand_subset(L, K)))
        age_list.append(0)

    for l in range(1, num):
        # reset G list, error list, and past_a list
        current_G_vals = []
        error_vals = []
        past_a_vals = current_a_vals.copy()

        # loop updating new 'a' value
        a_index = 0
        for L_k in L_k_list:
            G = np.zeros(n)
            error = 0
            for j in L_k:
                grad_out = EA_grad_phi(current_a_vals[a_index], X[:,j], C[1][j])
                G = G + grad_out[0]
                error = error + abs(grad_out[1])
            current_G_vals.append(G.copy())
            error_vals.append(error)
            if error < min_error:
                min_error = error
                best_a = current_a_vals[a_index].copy()
            current_a_vals[a_index] = current_a_vals[a_index] - eta * G/(K * (l**p))
            a_index += 1

        # returns index of a with minimum error (could be used for metrics)
        best_index = error_vals.index(min(error_vals))
        A[l] = past_a_vals[best_index]

        # evolving the population before next iteration - ages, G values, L_k list, a sets passed as parameters to selection
        num_selected = int(K / 2)

        selected = EA_selection(L_k_list, current_G_vals, age_list, current_a_vals, error_vals,
                                num_selected)  # need to define num_selected - number of selected members
        selected_L_k = selected[0]
        selected_ages = selected[1]
        selected_a = selected[2]

        after_breeding = EA_breed(selected_L_k, selected_ages, selected_a, m - 1, recombination_split,
                                  pop_size - len(selected_L_k))
        L_k_list = after_breeding[0]
        age_list = after_breeding[1]
        current_a_vals = after_breeding[2]


    after_breeding

    return A, best_a



# Efficient Evolutionary Stochastic Gradient Descent

#INPUTS:
#a0: initial guess for parameters a (can be randomly assigned or zero vector)
#C: coordinate data [x,y] x grouped and y grouped
#K: size of stochastic subset of data to gradient-step in
#eta: learning-rate/stepping size
#num: number of iterations/steps (need to account for size of population for fair comparison)
#pop_size: number of members for the evolving population
#recombination_split: split of genetic material to be acquired from either parent (e.g. 0.7 means 70% of genes from parent 1, 30% from parent 2)
#p: eta decay parameter

#OUTPUT:
#A: Matrix of a values throughout iteration



def EEvSGDdb(a0, C, eta, K, num, pop_size, recombination_split, p):
    # size of Operator
    m = np.size(C, 1)
    n = np.size(a0)
    X = F(C[0]) 

    # record best parameter set (in memory for easy access)
    A = np.ndarray((num, n))
    A[0] = a0

    # indices
    L = [i for i in range(m)]

    # initialize members of population (subset choices) & age list (all start at zero)
    L_k_list = []
    age_list = []
    for L_k_i in range(pop_size):
        L_k_list.append(np.sort(rand_subset(L, K)))
        age_list.append(0)

    for l in range(1, num):
        # reset G list, error list, and past_a list
        current_G_vals = []
        error_vals = []

        # loop updating new 'a' value
        a_index = 0
        for L_k in L_k_list:
            G = np.zeros(n)
            error = 0
            for j in L_k:
                grad_out = EA_grad_phi(A[l - 1], X[:,j], C[1][j])
                G = G + grad_out[0]
                error = error + abs(grad_out[1])
            current_G_vals.append(G.copy())
            error_vals.append(error)  # not sure whether list holds references or updated errors - if troubleshooting pay attn here

            a_index += 1

        # evolving the population before next iteration - ages, G values, L_k list, a sets passed as parameters to selection
        num_selected = int(K / 2)

        selected = EA_selection_singlea(L_k_list, current_G_vals, age_list, error_vals, num_selected)
        selected_L_k = selected[0]
        selected_ages = selected[1]
        selected_G = selected[2]

        after_breeding = EA_breed_singlea(selected_L_k, selected_ages, m - 1, recombination_split,
                                          pop_size - len(selected_L_k))
        L_k_list = after_breeding[0]
        age_list = after_breeding[1]
        avg_G = 0
        for g in selected_G:
            avg_G += g
        avg_G = avg_G/len(selected_G)
        A[l] = A[l - 1] - eta * avg_G / (K * (l ** p))


    after_breeding

    return A