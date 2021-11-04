import numpy as np
import scipy.linalg as la
from pymanopt.tools.multi import multilog
import time
def func(A,X):
    # A为(block，n，n)矩阵
    if A.ndim != 3:
        raise ValueError("A必须是3维")
    block = A.shape[0]
    X_inv = la.inv(X)
    X_inv_sq = la.sqrtm(X_inv)
    vec = multilog( np.einsum( 'mik,kj->mij',np.einsum('ik,mkj->mij',X_inv_sq,A),X_inv_sq) ,pos_def= True)
    distance_sum = np.einsum('pij,pji->',vec,vec)
    return 1/( 2*block)  *  distance_sum
def grad(A,X):
    if A.ndim != 3:
        raise ValueError("A必须是3维")
    block = A.shape[0]
    X_inv = la.inv(X)
    X_sq = la.sqrtm(X)
    X_inv_sq = la.sqrtm(X_inv)
    vec = multilog( np.einsum( 'mik,kj->mij',np.einsum('ik,mkj->mij',X_inv_sq,A),X_inv_sq) ,pos_def= True)
    vec_sum =  X_sq @ np.einsum('ijk->jk',vec) @  X_sq
    return -(1/( block)) * vec_sum 
def sum_f(A,X):
    if A.ndim != 4:
        raise ValueError("A必须是4维")
    n = A.shape[-1]
    return func(A.reshape(-1,n,n),X)

def sum_grad(A,X):
    if A.ndim != 4:
        raise ValueError("A必须是4维")
    n = A.shape[-1]
    return grad(A.reshape(-1,n,n),X)

