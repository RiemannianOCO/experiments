import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from core.online_problem import OnlineProblem  
from solver import *
from lib.function import frechet_mean_h
from experiment.fre_mean.hyp_100 import config

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



aver_value = np.zeros( T )
aver_time  = np.zeros( T )
rounds = 100
for _ in range(rounds):
    solver = OnlineZeroth()
    solver.optimize(ol_fre_prob, X_0)
    solver.calculate_aver_value() 
    solver.sum_time()
    aver_value += solver.aver_value_histories
    aver_time  += solver.time_sum
aver_value /= rounds
aver_time  /= rounds

np.save( fold_write  + 'time_ozo',aver_time)
np.save( fold_write  + 'data_ozo',aver_value)

'''
solver = OnlineBandit()
solver.optimize(os_problem,X_0,mul = 6)
solver.calculate_aver_value()

np.save( foldname + 'data_bandit',solver.aver_value_histories)
print('bandit solver completed')

#import os
#os.system('D:/ProgramData/Anaconda3/envs/Rieopt/python.exe d:/code/R-OCO/bin/frechet_mean/plot.py')

'''