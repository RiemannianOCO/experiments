import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from core.online_problem import OnlineProblem
from lib.function import pca
from solver.pos.online_two_bandit_pos import OnlineTwoPointBandit

import config

n=config.n
T=config.T
block=config.block
foldname = config.foldname
Gr =config.mfd

A = np.load(foldname + 'data_A.npy')
X_0 = config.X_0

ol_pca = OnlineProblem(     mfd = Gr,
                                data = A,
                                time = T,
                                param = config.param,
                                loss = pca.func,
                                grad = pca.grad,
                                ) 


aver_values = np.zeros(  T  )
aver_time = np.zeros(  T  )
solver = OnlineTwoPointBandit()
rounds = 1
for i in range(rounds):
    solver.optimize(ol_pca,X_0,mul = 1, mu=2)
    solver.calculate_aver_value()
    solver.sum_time()
    aver_values += solver.aver_value_histories
    aver_time   += solver.time_sum
aver_values = aver_values / rounds
aver_time = aver_time /rounds

np.save( foldname + 'data_two_bandit',aver_values)
np.save( foldname + 'time_two_bandit',aver_time)
print('2-bandit solver completed')



'''
list_mul = [0.01,0.1,1,10,20]
m = len(list_mul)
aver_values = np.zeros( (m,T) )
i = 0
solver = OnlineTwoPointBandit()
for mul in list_mul:
    solver.optimize(ol_problem,X_0,mul)
    solver.calculate_aver_value()
    aver_values[i] = solver.aver_value_histories
    i += 1 
    print('bandit solver completed',mul)
import matplotlib.pyplot as plt
for i in range(m):
    plt.plot( aver_values[i] ,label = list_mul[i])
plt.legend()
plt.show()

solver = OnlineTwoPointBandit()
solver.optimize(ol_problem,X_0,mul=1)
solver.calculate_aver_value()
import matplotlib.pyplot as plt
np.save( foldname + 'data_two_bandit',solver.aver_value_histories)
print('bandit solver completed')
'''