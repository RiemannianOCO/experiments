import numpy as np
from scipy import linalg as la
def zeta(kappa,D):
    if kappa >= 0:
        return 1
    temp = np.sqrt(-kappa) * D
    return temp / np.tanh(temp) 

def logdet(A):
    U = la.cholesky(A)
    y = 2*sum(np.log(np.diag(U)))
    return y

def sigma(K,D):
    if K<=0:
        return 0
    else:
        temp = np.sqrt(K) * D
        return max ( -temp / np.tan(temp) , 0 )