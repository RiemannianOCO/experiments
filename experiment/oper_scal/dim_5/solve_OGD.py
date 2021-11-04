import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from core.online_problem import OnlineProblem
from core.param import Param
from lib.function import oper_scal
from solver import *

import config

n=config.n
T=config.T
block=config.block
SPD =config.SPD

foldname = config.foldname
A = np.load(foldname + 'data_A.npy')
X_0 = config.X_0

os_problem = OnlineProblem(     mfd = SPD,
                                data = A,
                                time = T,
                                param= config.param,
                                loss = oper_scal.func,
                                grad = oper_scal.grad,
                                ) 



solver = OnlineGradientDescent()
solver.optimize(os_problem,X_0)
solver.calculate_aver_value()
solver.sum_time()
np.save( foldname+ 'data_gradient',solver.aver_value_histories)
np.save( foldname + 'time_gradient',solver.time_sum)
print('gradient solver completed')
