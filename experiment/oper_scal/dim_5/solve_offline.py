import sys
from typing import List
sys.path.append('.')

import numpy as np
from pymanopt.manifolds import PositiveDefinite
from core.offline_problem import OfflineProblem
from solver.offline_solver import OfflineSolver
from pymanopt.solvers import ConjugateGradient
from lib.function import oper_scal
import config

n=config.n
T=config.T
block=config.block
foldname = config.foldname

list_T = list(range(0,T,200))
list_T.append(T-1
)
SPD = PositiveDefinite(n, k=1)
A = np.load( foldname + 'data_A.npy' )
X_0 = np.eye(n)

off_prob = OfflineProblem(  mfd = SPD,
                            data = A,
                            time = T,
                            loss= oper_scal.func,
                            grad= oper_scal.grad
                            )




solver = OfflineSolver(solver=ConjugateGradient,mingrad = 1e-3)
X_0 = solver.optimize(off_prob,X_0,[T-1])
fin_value = solver.offline_histories[0]

solver.optimize(off_prob,X_0,list_T)

np.save( foldname +'data_offline',solver.offline_histories)
np.save( foldname + 'list_T',list_T)
print('offline solver completed')