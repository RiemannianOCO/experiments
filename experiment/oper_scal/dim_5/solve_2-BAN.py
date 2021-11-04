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

aver_values = np.zeros(  T  )
aver_time = np.zeros(  T  )
solver = OnlineTwoPointBandit()
rounds = 100
for i in range(rounds):
    solver.optimize(os_problem,X_0,mul = 1)
    solver.calculate_aver_value()
    solver.sum_time()
    aver_values += solver.aver_value_histories
    aver_time   += solver.time_sum
    print(i)
aver_values = aver_values / rounds
aver_time = aver_time /rounds

np.save( foldname + 'data_two_bandit',aver_values)
np.save( foldname + 'time_two_bandit',aver_time)
print('2-bandit solver completed')
