import numpy as np

#INPUTS:
#L: list to randomly choose subset from
#k: k sized subset tp pick from array
#OUTPUT:
#subset: the random subset array picked from original list
def rand_subset(L, k):
    s = len(L)
    items = np.linspace(0,s-1,s).astype(int)
    np.random.shuffle(items)
    index = items[:k]
    L = np.array(L)
    subset = L[index]
    return subset

#INPUTS:
# ls: list of items to select from
# stn: array of strength's corresponding to items
# n: top n to return
#OUTPUT:
# select: selection of n most fit items
def selection(ls, stn, n):
    stn = -stn
    index = np.argsort(stn)[:n]
    select = ls[index]
    return select

#INPUTS:
# subset: array of old polulation
# L: list of all possible people in population
# k: size of subset, the old population 
# n: number of top people kept from last population in subset
#OUTPUT:
# new_pop: new iteration of list including
#           elements of last list and and randomized
#           rest of subset population.
def repopulate(subset,L, k, n):
    #amount r needed to repopulate
    r = k - n
    #remove elements already included
    remain = np.setdiff1d(L,subset)
    # pick r new people
    rest = rand_subset(remain, r)
    #new people join old population
    new_pop = np.concatenate((subset, rest))
    return new_pop

#Imagine minimization problem: F*a=y (or a*f(x)=y, a linear parameter)
#Find the a that gets F*a closest to y
#F is mxn matrix with columns populated
#by n different f_j(x) functions or features
#a is n-vector, y is an m-vector
#We are optimizing over a's 
#but stochastic over the choice i's,
#the different row of equations
# which is also the equivalent number of loss fucntions

#Possible function space 
#(create your own, make sure to update both)

#INPUTs:
# a: parameters needing optimization (size n), use numpy array
# x: 1d positional scalar (lowercase)
#   (may have n variational functionals f_j(x) )
#   i.e, a_0*f_0(x)+a_1*f_1(x)+...+a_n*f_n(x)
#OUTPUT:
# X: feature F(x) values 
def F(x):
    X = np.array([x**0,x,x**2,x**3])
    return X


#INPUTS:
# X: F(x) values after function action
# a: parameters needing optimization (size n), use numpy array
#OUTPUT:
# f: function value evaluated at 'a' and 'x' (lowercase)
def f(a,X) :
    f = np.dot(a,X)
    return f

#Exact gradient of f(a,x), linear in a, in respect to 'a'
#INPUTS:
# X: F(x) values after function action
# a: parameters needing optimization (size n), use numpy array
def grad_f(a,X):
    return X

#Possible optional generator of data y
#Instead import your own data
#INPUTS:
# a: linear factors (numpy array)
# x: x values in F(x) (scalar or numpy array)
# s: variance/size of noise to add
# norm: boolian determines variance dependence on size of f(a,X)
def Y(a,x,s = 1, norm = False):
    X = F(x)
    m = x.size
    if norm == False:
        return np.array([x, f(a,X) + np.random.normal(0,s,m)])
    else:
        f0 = f(a,X)
        nor = np.zeros(m)
        for i in range(m):
            nor[i] = np.absolute(f0[i])
        return np.array([x,f0 + nor*np.random.normal(0,s,m)])
            
#Example loss function to be optimized
#INPUTS:
# a: array of linear coefficients
# X: F(x) values after function action
# y: scalar of coordiante 
def phi(a,X,y):
    return (f(a,X) - y)**2

#Given phi loss function above
#Local Gradient
def grad_phi(a,X,y):

    return 2*(f(a,X)-y)*grad_f(a,X)

#Total loss function (an type of average)
def cost(X,y,a):
    P = phi(a,X,y)
    return np.average(P)


 #NON-Linear Rational
#Possible optional generator of data y
#Instead import your own data
#INPUTS:
# a: linear factors numerator (numpy array)
# b: linear factors denominator (numpy array)
# x: x values in F(x) (scalar or numpy array)
# s: variance/size of noise to add
# norm: boolian determines variance dependence on size of f(a,X)
def Y2(a,b,x,s = 1, norm = False):
    X = F(x)
    m = x.size
    if norm == False:
        return np.array([x, f(a,b,X) + np.random.normal(0,s,m)])
    else:
        f0 = f(a,b,X)
        nor = np.zeros(m)
        for i in range(m):
            nor[i] = np.absolute(f0[i])
        return np.array([x,f0 + nor*np.random.normal(0,s,m)])

#INPUTS:
# X: F(x) values after function action
# a: parameters needing optimization (size n), use numpy array
#OUTPUT:
# f: function value evaluated at 'a' and 'x' (lowercase)
def f2(a,b,X) :
    f2 = np.dot(a,X)/np.dot(b,X)
    return f2

#Exact gradient of f(a,x), linear in a, in respect to 'a'
#INPUTS:
# X: F(x) values after function action
# a: parameters needing optimization (size n), use numpy array
def grad_f2(a,b,X):
    s1 = a.size
    s2 = b.size
    grad = np.zeros(s1+s2)
    grad[:s1] = X/np.dot(a,X)
    grad[s1:] = -X/(np.dot(a,X)**2)
    return grad

#Example loss function to be optimized
#INPUTS:
# a: array of linear coefficients
# X: F(x) values after function action
# y: scalar of coordiante 
def phi2(a,X,y):
    return (f(a,b,X) - y)**2

#Given phi loss function above
#Local Gradient
def grad_phi2(a,b,X,y):

    return 2*(f(a,X)-y)*grad_f(a,X)

#Total loss function (an type of average)
def cost2(X,y,a):
    P = phi2(a,X,y)
    return np.average(P)

#INPUT:
# X: input values 'x'
# s: size of features
#OUTPUT:
# Fu: Function/Feature values standardized
# m: array of mean shifts for each feature
# v_s: array of standard deviation scales for each feature
def preprocess(X,s):
    m = np.mean(X, axis = 1)
    v = np.var(X, axis = 1)
    v_s = np.sqrt(v)
    for i in range(1,s):
        X[i] = (X[i]-m[i])
    if (v[i] != 0):
        X[i] = X[i]/v_s[i]
    return X, m, v_s

#INPUT:
#a: preprocessed learned 'a' values
#m: array of mean shifts for each feature
#v_s: array of standard deviation scales for each feature
#s: size of features
#OUTPUT:
#a_r: augmented 'a' values that correspond to original problem
def postprocess(a,m,v_s,s):
    a_r = np.zeros(s)
    for i in range(1,s):
        a_r[i] = a[i]/v_s[i]
    a_r[0]= a[0]-np.sum(a[1:]*m[1:]/v_s[1:])
    return a_r