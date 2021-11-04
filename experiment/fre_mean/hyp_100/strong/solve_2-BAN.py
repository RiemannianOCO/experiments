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

A = np.load( fold_read + 'data_A.npy')
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
    solver = OnlineTwoPointBandit()
    solver.optimize(ol_fre_prob, X_0, mul = 1, mu=1, setoff=170)
    solver.calculate_aver_value() 
    solver.sum_time()
    aver_value += solver.aver_value_histories
    aver_time  += solver.time_sum
aver_value /= rounds
aver_time  /= rounds

np.save( fold_write + 'time_two_bandit',aver_time)
np.save( fold_write + 'data_two_bandit',aver_value)
print('2-bandit solver completed')












'''
import matplotlib.pyplot as plt

plt.plot( solver.aver_value_histories ,label = 'C')
plt.legend()
plt.show()



list_mul = [0.05,0.1,1,10,20,30,40]
m = len(list_mul)
aver_values = np.zeros( (m,T) )
i = 0
solver = OnlineTwoPointBandit()
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

solver = OnlineTwoPointBandit()
solver.optimize(ol_fre_prob_sc,X_0,mul = 8)
solver.calculate_aver_value()
solver.sum_time()

solver2 = OnlineTwoPointBandit()
solver2.optimize(ol_fre_prob,X_0,mul = 8)
solver2.calculate_aver_value()
solver2.sum_time()

import matplotlib.pyplot as plt
plt.loglog( solver2.aver_value_histories ,label = 'C')
plt.loglog( solver.aver_value_histories ,label = 'SC')
plt.legend()
plt.show()

np.save( foldname + 'time_two_bandit',solver.time_sum)
np.save( foldname + 'data_two_bandit',solver.aver_value_histories)
print('2-bandit solver completed')
'''


