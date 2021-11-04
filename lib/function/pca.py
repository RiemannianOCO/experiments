import numpy as np
from pymanopt.manifolds import stiefel
from pymanopt.manifolds.stiefel import Stiefel
import scipy.linalg as la

def func(A, X):
    # A为(block，N)矩阵
    block = A.shape[0]
    sum = - 0.5 * np.trace(A @ X @ X.T @ A.T) 
    return sum/block

def grad(A, X):
    block = A.shape[0]
    N = X.shape[0]
    Egrad = - np.einsum('ij,ik->jk',A,A) @ X
    Rgrad = (np.eye(N) - X @ X.T) @ Egrad 
    return Rgrad/block


def sum_f( A , X ):
    N = A.shape[-1]
    return func(A.reshape(-1,N),X) 


def sum_grad( A , X ):
    N = A.shape[-1]
    return grad(A.reshape(-1,N),X) 

