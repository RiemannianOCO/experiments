import numpy as np
import scipy.linalg as la
from pymanopt.manifolds import PositiveDefinite

def func(A,X):

    # A为(block，n，n)矩阵
    if A.ndim != 3:
        raise ValueError("A必须是3维")
    block = A.shape[0]
    n = A.shape[1]
    SPD = PositiveDefinite(n, k=1)
    value = 0
    for i in range(block):
        value = value + ( 1 /(2 * block) ) * (SPD.dist(A[i],X))**2
    return value

def grad(A,X):
    if A.ndim != 3:
        raise ValueError("A必须是3维")
    block = A.shape[0]
    n = A.shape[1]
    SPD = PositiveDefinite(n, k=1)

    grad = np.zeros((n,n)) 
    for i in range(block):
        grad = grad - 1/block * ( SPD.log(X,A[i]) )
    return grad