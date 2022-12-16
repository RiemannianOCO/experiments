import sys
from typing import List
sys.path.append('.')

import autograd.numpy as np
from pymanopt.manifolds import SymmetricPositiveDefinite
from core.offline_problem import OfflineProblem
from pymanopt import Problem,function
from pymanopt.optimizers import *
from lib.function import oper_scal
import config
import pymanopt

n=config.n
T=config.T
block=config.block
foldname = config.foldname

SPD = SymmetricPositiveDefinite(n, k=1)
A = np.load( foldname + 'data_A.npy' )
Y = SPD.random_point()
X_0 = np.eye(n)
mfd = SPD
max_res = -1
for i in range(10):
    print(i)
    @pymanopt.function.autograd(mfd)
    def cost(x):
        sum = np.zeros(x.shape)
        for a in A[i]:
            sum = sum + a @ x @ a.T
        return np.log(np.linalg.det(sum))
    for _ in range(100):
        x_0 = SPD.random_point()
        for _ in range(100):
            v = SPD.random_tangent_vector(x_0)
            ehess = cost.get_hessian_operator()
            egrad = cost.get_gradient_operator()
            rhess = SPD.euclidean_to_riemannian_hessian(
                point=x_0,
                euclidean_gradient= egrad(x_0),
                euclidean_hessian=ehess(x_0,v),
                tangent_vector= v
            )

            res = (SPD.inner_product(x_0,rhess,v)) **0.5 
            if max_res < res:
                max_res = res


print(max_res)