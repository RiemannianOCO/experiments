import numpy as np
import scipy.linalg as la

def func(A,X):
    return -0.5 * ( X @ ( A @ X ) ) 

def grad(A,X):
    egrad = - A @ X
    grad = egrad - X @ X.T @ egrad 
    return grad