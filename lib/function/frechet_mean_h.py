import numpy as np
def mdot(X,Y):
    Y_m  = np.copy(Y)
    if len(Y_m.shape) == 1:
        Y_m[-1] = -Y_m[-1]
    else:
        Y_m[: , -1] = -Y_m[: , -1]
    return np.dot(Y_m,X)

def func(A,X):
    block = A.shape[0]
    return (1/ (2 * block)) * np.sum (np.arccosh( -mdot(X,A)) ** (2))

def grad(A,X):
    block = A.shape[0]
    m_dot= mdot( X , A )
    dist = np.arccosh  ( -m_dot )
    vec = A + np.outer(m_dot,X)
    log_vec =   ((dist * ( ( m_dot**2 -1 ) ** -(0.5)))  * vec.T).T



    return - (1 / block ) *  log_vec.sum(axis = 0)


def sum_f(A,X):
    n = A.shape[-1]
    return func(A.reshape(-1,n),X)

def sum_grad(A,X):
    n = A.shape[-1]
    return grad(A.reshape(-1,n),X)
