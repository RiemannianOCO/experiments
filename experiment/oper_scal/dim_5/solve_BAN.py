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
foldname = config.foldname
SPD =config.SPD


A = np.load(foldname + 'data_A.npy')
X_0 = config.X_0

os_problem = OnlineProblem(     mfd = SPD,
                                data = A,
                                time = T,
                                param= config.param,
                                loss = oper_scal.func,
                                grad = oper_scal.grad,
                                ) 

values = []
time = []
solver = OnlineBandit()
rounds = 100
for i in range(rounds):
    solver.optimize(os_problem,X_0,mul = 10)
    solver.calculate_aver_value()
    solver.sum_time()
    values.append(solver.aver_value_histories)
    time.append(solver.time_sum)
    print(i)
arr_values = np.array(values)
arr_time = np.array(time)
aver_values , std_values = np.mean(arr_values,axis=0) , np.std(arr_values,axis=0)
aver_time = np.mean(arr_time,axis=0)
np.save( foldname + 'data_bandit',aver_values)
np.save( foldname + 'std_bandit',std_values)
np.save( foldname + 'time_bandit',aver_time)
print('bandit solver completed')