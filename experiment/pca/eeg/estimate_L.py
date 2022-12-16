import sys
from typing import List
sys.path.append('.')

import autograd.numpy as np
from pymanopt import Problem,function
from pymanopt.optimizers import *
from lib.function import pca
import config
import pymanopt

n=config.n
T=config.T
block=config.block
foldname = config.foldname
A = np.load( foldname + 'data_A.npy' )

mfd = config.mfd
max_res = -1
for i in range(10):
    print(i)
    @pymanopt.function.autograd(mfd)
    def cost(x):
        block = A[i].shape[0]
        sum = - 0.5 * np.trace(A[i] @ x @ x.T @ A[i].T) 
        return sum /block
    for _ in range(100):
        x_0 = mfd.random_point()
        for _ in range(100):
            v = mfd.random_tangent_vector(x_0)
            ehess = cost.get_hessian_operator()
            egrad = cost.get_gradient_operator()
            rhess = mfd.euclidean_to_riemannian_hessian(
                point=x_0,
                euclidean_gradient= egrad(x_0),
                euclidean_hessian=ehess(x_0,v),
                tangent_vector= v
            )

            res = (mfd.inner_product(x_0,rhess,v)) 
            if max_res < res:
                max_res = res
        print(max_res **0.5 )

