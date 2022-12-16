import sys
from typing import List
sys.path.append('.')

import numpy as np
from pymanopt.manifolds import SymmetricPositiveDefinite
from core.offline_problem import OfflineProblem
from pymanopt import Problem,function
from pymanopt.optimizers import *
from lib.function import frechet_mean_h
import config

n=config.n
T=config.T
block=config.block
foldname = config.foldname

mfd = config.mfd
A = np.load( foldname + 'data_A.npy' )
X_0 = mfd.center

X=np.zeros((T,*X_0.shape))
solver = SteepestDescent(min_gradient_norm = 1e-3)
solver._verbosity = 0
for t in range(T):
    @function.numpy(mfd)
    def func(X):
        return frechet_mean_h.func(A[t],X)

    @function.numpy(mfd)
    def grad(X):
        return frechet_mean_h.grad(A[t],X)

    off_problem = Problem(manifold = mfd, cost=func, riemannian_gradient=grad)
    X[t] = solver.run(off_problem,initial_point =X_0).point
    if t>0:
        print(mfd.dist(X[t],X[t-1]))

