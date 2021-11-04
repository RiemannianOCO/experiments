import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from core.online_problem import OnlineProblem
from core.param import Param
from lib.function import pca
from solver import *

import config
n=config.n
T=config.T
block=config.block
foldname = config.foldname
Gr =config.mfd


A = np.load(foldname + 'data_A.npy')
X_0 = config.X_0

ol_problem = OnlineProblem(     mfd = Gr,
                                data = A,
                                time = T,
                                param = config.param,
                                loss = pca.func,
                                grad = pca.grad,
                                ) 

solver = OnlineGradientDescent()
solver.optimize(ol_problem,X_0,mu=2)
solver.calculate_aver_value()
solver.sum_time()

np.save( foldname + 'time_gradient',solver.time_sum)
np.save( foldname+ 'data_gradient',solver.aver_value_histories)
print('gradient solver completed')