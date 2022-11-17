import os
import sys

import numpy as np
from pymanopt.optimizers import *
sys.path.append(os.getcwd())

from core.offline_problem import OfflineProblem
from lib.function import frechet_mean_h
from solver.offline_solver import OfflineSolver

import config

n=config.n
T=config.T
block=config.block
foldname = config.foldname
Hn = config.mfd
A = np.load( foldname + 'data_A.npy' )
X_0 = Hn.center


grid_num = 50
list_T= list(range(0,T,50))
list_T.append(T-1)

off_prob = OfflineProblem(  mfd = Hn,
                            data = A,
                            time = T,
                            sum_loss= frechet_mean_h.sum_f,
                            sum_grad= frechet_mean_h.sum_grad
                            )


solver = OfflineSolver(solver = SteepestDescent, mingrad = 1e-3)
solver.optimize(off_prob,X_0,list_T)

np.save( foldname +'data_offline',solver.offline_histories)
np.save( foldname + 'list_T',list_T)
print('offline solver completed')
