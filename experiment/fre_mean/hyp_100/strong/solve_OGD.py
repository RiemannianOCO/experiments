import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from core.online_problem import OnlineProblem
from experiment.fre_mean.hyp_100 import config
from lib.function import frechet_mean_h
from solver import *

n=config.n
T=config.T
block=config.block
Hn =config.mfd

fold_read = config.foldname
fold_write = config.fold_strong

A = np.load(fold_read + 'data_A.npy')
X_0 = config.X_0

ol_fre_prob = OnlineProblem(    mfd = Hn,
                                data = A,
                                time = T,
                                param = config.param,
                                loss = frechet_mean_h.func,
                                grad = frechet_mean_h.grad,
                                ) 

solver = OnlineGradientDescent()
solver.optimize(ol_fre_prob,X_0,mu=1)
solver.calculate_aver_value()
solver.sum_time()

np.save( fold_write + 'time_gradient_sc',solver.time_sum)
np.save( fold_write + 'data_gradient_sc',solver.aver_value_histories)
print('gradient solver completed')
