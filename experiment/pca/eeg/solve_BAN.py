import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from core.online_problem import OnlineProblem
from lib.function import pca
from solver.pos.online_bandit_pos import OnlineBandit

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
solver = OnlineBandit()
rounds = 100
for i in range(rounds):
    solver.optimize(ol_pca,X_0,mul = 1,mu=0.1)
    solver.calculate_aver_value()
    solver.sum_time()
    aver_values += solver.aver_value_histories
    aver_time   += solver.time_sum
    print(i)
aver_values = aver_values / rounds
aver_time = aver_time /rounds
np.save( foldname + 'data_bandit',aver_values)
np.save( foldname + 'time_bandit',aver_time)
print('bandit solver completed')





'''
list_delta = [0.001,0.01,0.1,1,10]
list_alpha = [0.001,0.01,0.1,1,10]

m1 = len(list_delta)
m2 = len(list_alpha)
aver_values = np.zeros( (m1,m2,T) )

solver = OnlineBanditTest()
for i in range(m1):
    for j in range(m2):
        delta_i = list_delta[i]
        alpha_j = list_alpha[j]
        solver.optimize(ol_problem,X_0,delta = delta_i, alpha=alpha_j)
        solver.calculate_aver_value()
        aver_values[i,j] = solver.aver_value_histories
        print('bandit solver completed',delta_i,alpha_j)

import matplotlib.pyplot as plt

for i in range(m1):
    for j in range(m2):
        delta_i = list_delta[i]
        alpha_j = list_alpha[j]
        plt.plot( aver_values[i,j] ,label = ('d:{}+a:{}').format(delta_i,alpha_j))
plt.legend()
plt.show()


solver = OnlineBanditTest()
solver.optimize(ol_problem,X_0,delta = 0.8, alpha=0.02)
solver.calculate_aver_value()

np.save( foldname + 'data_bandit',solver.aver_value_histories)
print('bandit solver completed')
'''
