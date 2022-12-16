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

foldname = config.foldname

A = np.load(foldname + 'data_A.npy')
X_0 = config.X_0

ol_fre_prob = OnlineProblem(    mfd = Hn,
                                data = A,
                                time = T,
                                param = config.param,
                                loss = frechet_mean_h.func,
                                grad = frechet_mean_h.grad,
                                ) 



values = []
time = []
rounds = 100
for i in range(rounds):
    solver = OnlineZeroth()
    solver.optimize(ol_fre_prob, X_0,sigma=1,L=config.param.zeta,V=1)
    solver.calculate_aver_value() 
    solver.sum_time()
    values.append(solver.aver_value_histories)
    time.append(solver.time_sum)
    print(i)
arr_values = np.array(values)
arr_time = np.array(time)
aver_values , std_values = np.mean(arr_values,axis=0) , np.std(arr_values,axis=0)
aver_time = np.mean(arr_time,axis=0)

np.save( foldname + 'data_ozo',aver_values)
np.save( foldname + 'std_ozo',std_values)
np.save( foldname + 'time_ozo',aver_time)

'''
solver = OnlineBandit()
solver.optimize(os_problem,X_0,mul = 6)
solver.calculate_aver_value()

np.save( foldname + 'data_bandit',solver.aver_value_histories)
print('bandit solver completed')

#import os
#os.system('D:/ProgramData/Anaconda3/envs/Rieopt/python.exe d:/code/R-OCO/bin/frechet_mean/plot.py')

'''