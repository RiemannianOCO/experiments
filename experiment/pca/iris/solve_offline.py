import os
import sys

import numpy as np
from pymanopt.optimizers import ConjugateGradient

sys.path.append(os.getcwd())
from core.offline_problem import OfflineProblem
from lib.function import pca
from solver.offline_solver import OfflineSolver

import config

n=config.n
T=config.T
block=config.block
foldname = config.foldname
Gr = config.mfd
A = np.load( foldname + 'data_A.npy' )
X_0 = config.X_0

list_T= list(np.arange(config.T))


off_prob = OfflineProblem(  mfd = Gr,
                            data = A,
                            time = T,
                            sum_loss = pca.sum_f,
                            sum_grad = pca.sum_grad 
                        ) 

solver = OfflineSolver(solver = ConjugateGradient, mingrad = 1e-3)

X_0 = solver.optimize(off_prob,X_0,[T-1])
solver.optimize(off_prob,X_0,list_T)


np.save( foldname +'data_offline',solver.offline_histories)
np.save( foldname + 'list_T',list_T)
print('offline solver completed')