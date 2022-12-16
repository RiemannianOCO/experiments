import sys
from typing import List
sys.path.append('.')

import numpy as np
from pymanopt.manifolds import SymmetricPositiveDefinite
from core.offline_problem import OfflineProblem
from pymanopt import Problem,function
from pymanopt.optimizers import *
from lib.function import oper_scal
import config

n=config.n
T=config.T
block=config.block
foldname = config.foldname
dis =-1
SPD = SymmetricPositiveDefinite(n, k=1)
A = np.load( foldname + 'data_A.npy' )
X_0 = np.eye(n)
res = -1
dis = 0
mfd = SPD
X=np.zeros((T,n,n))
solver = ConjugateGradient(min_gradient_norm = 1e-3)
solver._verbosity = 0
for t in range(T):
    @function.numpy(mfd)
    def func(X):
        return oper_scal.func(A[t],X)

    @function.numpy(mfd)
    def grad(X):
        return oper_scal.grad(A[t],X)

    off_problem = Problem(manifold = mfd, cost=func, riemannian_gradient=grad)
    X[t] = solver.run(off_problem,initial_point =X_0).point
    if t>0:
        dis = mfd.dist(X[t],X[t-1])
    if dis> res:
        res = dis
    print(res)