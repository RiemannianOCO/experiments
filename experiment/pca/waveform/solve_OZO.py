import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from core.online_problem import OnlineProblem
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

ol_pca = OnlineProblem(     mfd = Gr,
                                data = A,
                                time = T,
                                param = config.param,
                                loss = pca.func,
                                grad = pca.grad,
                                ) 

solver = OnlineZeroth()
rounds = 10
values = []
time = []
for i in range(rounds):
    solver.optimize(ol_pca,X_0,sigma=10,L=20)
    solver.calculate_aver_value()
    values.append(solver.aver_value_histories)
    time.append(solver.time)
    print(i)
arr_values = np.array(values)
arr_time = np.array(time)
print(arr_values[:,0])
print(arr_values[:,-1])
aver_values , std_values = np.mean(arr_values,axis=0) , np.std(arr_values,axis=0)
aver_time = np.mean(arr_time,axis=0)
np.save( foldname + 'data_ozo',aver_values)
np.save( foldname + 'std_ozo',std_values)
np.save( foldname + 'time_ozo',aver_time)
print('zero solver completed')





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
