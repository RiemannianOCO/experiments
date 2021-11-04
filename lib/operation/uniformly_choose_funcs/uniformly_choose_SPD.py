import numpy as np
from scipy import linalg as la


def uniformly_choose_SPD(X, delta, n):
    v = delta * choose_on_sphere(n)
    X_sqrt = la.sqrtm(X)
    u = X_sqrt @ v @ X_sqrt
    return u
    '''
     while True:
    
        v = delta * choose_on_sphere(n)
        thre = np.random.rand()
        ratio = density(v) / 1.85
        if ( ratio <= thre ):
            u = X_sqrt @ v @ X_sqrt
    
    '''

def density(Q):
    kappa = diag_curv(Q)
    n = Q.shape[0]
    mul = 1
    for i in range(n):
        for j in range(i + 1):
            if kappa[i, j] != 0.0:
                mul = mul * np.sinh(np.sqrt(-kappa[i, j])) / (np.sqrt(-kappa[i, j]))
    return mul


def diag_curv(Q, return_frame=False):
    n = Q.shape[0]
    dim = int(n * (n + 1) / 2)
    [w, v] = la.eig(Q)
    frame = np.zeros((n, n, n, n))
    kappa = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            v_i = v[:, i].reshape(-1, 1)
            v_j = v[:, j].reshape(-1, 1)
            if i == j:
                frame[i, j] = v_i @ v_j.T
            else:
                frame[i, j] = (1 / np.sqrt(2)) * (v_i @ v_j.T + v_j @ v_i.T)
            kappa[i, j] = -(1 / 4) * (w[i] - w[j]) ** 2
    if return_frame:
        return frame, kappa
    else:
        return kappa


def choose_on_sphere(n):
    dim = int(n * (n + 1) / 2)
    vec = np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    mat_diag = np.diag(vec[:n])
    mat_upper = np.zeros((n, n))
    mat_upper[np.triu_indices(n, k=1)] = vec[n:] / np.sqrt(2)
    mat = mat_diag + mat_upper + mat_upper.T
    return mat

