import os
import sys

import numpy as np
from pymanopt.solvers import *
sys.path.append(os.getcwd())

from core.offline_problem import OfflineProblem
from lib.function import frechet_mean_h
from solver.offline_solver import OfflineSolver

import config

n=config.n
T=config.T
block=config.block
foldname = config.foldname
mfd = config.mfd
A = np.load( foldname + 'data_A.npy' )

max_d = -1
max_g = -1
for _ in range(100):
    X_0 = mfd.rand()
    X_0 = mfd.exp( mfd.center , 5  * mfd.log(mfd.center,X_0) / mfd.dist(mfd.center,X_0) )
    for i in range(T):
        max_d = max( frechet_mean_h.func( A[i],X_0 ) , max_d   )
        max_g = max( mfd.norm(X_0,frechet_mean_h.grad( A[i],X_0 )) , max_g   )
print(max_d,max_g)