from utils import *
from EA_utils import *
from benchmark import SGDdb
from models import *
import time



def test_EASGD(C,num,pop_size,choice):
    a0 = np.array([1, 1, 1, 1])
    num_points = 40
    K = 10
    if choice==1:
        eta = 0.000009
        p = 0.07
        A = EEvSGDdb(a0, C, eta, K, num, pop_size,0.7,p) 
    elif choice==2:
        eta = 0.00000015
        p = 0
        A = CEvSGDdb(a0, C, eta, K, num, pop_size,0.7,p) 
    elif choice==3:
        eta = 0.0000081
        p = 0.07
        A = SSGDdb(a0,C,eta,K,num,p,5)
    else:
        eta = 0.0000081
        p = 0.07
        A = SGDdb(a0,C,eta,K,num,p)

    return A


def test_runtimes():
    a_true = np.array([1,-1,0.1,0.01])
    M = Y(a=a_true,x=np.array([((20/40)*i-10) for i in range(40)]), s = 0, norm = False)

    SSGD_time = time.time()
    for i in range(5):
        SSGD = test_EASGD(M,100000,4,3)
    SSGD_time = time.time() - SSGD_time

    EEvSGD_time = time.time()
    for i in range(5):
        EEvSGD = test_EASGD(M,100000,4,1)
    EEvSGD_time = time.time() - EEvSGD_time

    CEvSGD_time = time.time()
    for i in range(5):
        CEvSGD = test_EASGD(M,100000,4,2)
    CEvSGD_time = time.time() - CEvSGD_time

    SGD_time = time.time()
    for i in range(5):
        SGD = test_EASGD(M,100000,4,4)
    SGD_time = time.time() - SGD_time

    return [SSGD_time, EEvSGD_time, CEvSGD_time, SGD_time]

if __name__ == "__main__":
    test_runtimes()