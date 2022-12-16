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
                                grad = frechet_mean_h.grad
                            )
values = []
time = []
rounds = 100
for _ in range(rounds):
    solver = OnlineBandit()
    solver.optimize(ol_fre_prob,X_0,mul = 7,mu=1, setoff= 2125 )
    solver.calculate_aver_value() 
    solver.sum_time()
    values.append(solver.aver_value_histories)
    time.append(solver.time_sum)
arr_values = np.array(values)
arr_time = np.array(time)
aver_values , std_values = np.mean(arr_values,axis=0) , np.std(arr_values,axis=0)
aver_time = np.mean(arr_time,axis=0)
np.save( foldname + 'data_bandit',aver_values)
np.save( foldname + 'std_bandit',std_values)
np.save( foldname + 'time_bandit',aver_time)
print('bandit solver completed')



'''
list_mul = [0.1,1,4,6,8,10,15]
m = len(list_mul)
aver_values = np.zeros( (m,T) )
i = 0
solver = OnlineBandit()
for mul in list_mul:
    solver.optimize(ol_fre_prob,X_0,mul)
    solver.calculate_aver_value()
    aver_values[i] = solver.aver_value_histories
    i += 1
    print('bandit solver completed',mul)
import matplotlib.pyplot as plt

for i in range(m):
    #plt.loglog( aver_values[i] ,label = list_mul[i])
    plt.plot( aver_values[i] ,label = list_mul[i])
plt.legend()
plt.show()
'''

'''
solver = OnlineBandit()
solver.optimize(ol_fre_prob_sc,X_0,mul = 8)
solver.calculate_aver_value()
solver.sum_time()

solver2 = OnlineBandit()
solver2.optimize(ol_fre_prob,X_0,mul = 8)
solver2.calculate_aver_value()
solver2.sum_time()

import matplotlib.pyplot as plt

plt.loglog( solver2.aver_value_histories ,label = 'C')
plt.loglog( solver.aver_value_histories ,label = 'SC')
plt.legend()
plt.show()

'''