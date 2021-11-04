import numpy as np
import scipy.linalg as la
from .std_func import logdet

def func(A,X):
    # A为(block，n，n)矩阵
    if A.ndim != 3:
        raise ValueError("A必须是3维")
    value = logdet ( func_T(A,X) )  - logdet(X)
    return value

def grad(A,X):
    return grad_logdet_T(A,X)-X


def func_T(A,X):
    if A.ndim != 3:
        raise ValueError("A必须是3维")
    block = A.shape[0]
    n = A.shape[1]
    sum = np.zeros( (n,n) ) 
    for i in range(block):
        sum = sum + A[i] @ X @ A[i].T
    return sum  

def grad_logdet_T(A,X):
    block = A.shape[0]
    dim = A.shape[1]
    T_inv = la.inv(func_T(A,X))
    grad = np.zeros( (dim,dim) )
    for i in range(block):
        grad = grad + A[i].T @ T_inv @ A[i]
    grad  = X @ grad @ X
    return grad
'''

    def range_func(self,X,time_range): #for debug
        value = 0
        for i in time_range:
            value = value + self.func(X,i)
        return value
''' 
    