import os
import sys

import numpy as np
from pymanopt.optimizers import ConjugateGradient

sys.path.append(os.getcwd())
from core.offline_problem import OfflineProblem
from lib.function import pca
from solver.offline_solver import OfflineSolver

import config
from sklearn import decomposition
n=config.n
T=config.T
block=config.block
foldname = config.foldname
Gr = config.mfd
A = np.load( foldname + 'data_A.npy' )
sol_X = []
sol_value = []

list_T= list(np.arange(100,config.T,100))

list_T.append(T-1)

X = np.copy(A[0]).reshape((-1,14))
sol_value.append ( -0.5 * np.sum(X**2) )
print(-0.5 * np.sum(X**2))


pca = decomposition.PCA(n_components=3)
for i in list_T:
    print(i)
    X = np.copy(A[:i]).reshape((-1,14))
    pca.fit(X)
    Y = pca.transform(X)
    func = - 0.5 * np.trace( Y @ Y.T) / (i)
    sol_value.append( func )
    print(func)

np.save( foldname +'data_offline',sol_value)
list_T.insert(0,0)
np.save( foldname + 'list_T',list_T)

