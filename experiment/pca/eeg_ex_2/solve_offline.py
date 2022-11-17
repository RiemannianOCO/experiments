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
X = config.X

list_T= list(np.arange(0,config.T-1,100))
print(list_T[0])
list_T.append(T-1)


off_prob = OfflineProblem(  mfd = Gr,
                            data = A,
                            time = T,
                            sum_loss = pca.sum_f,
                            sum_grad = pca.sum_grad 
                        ) 

solver = OfflineSolver(solver = ConjugateGradient, mingrad = 1e-3)
#solver.optimize(off_prob,X,[T-1])
solver.optimize(off_prob,X,[0])


#np.save( foldname +'data_offline',solver.offline_histories)
#np.save( foldname + 'list_T',list_T)
print('offline solver completed')