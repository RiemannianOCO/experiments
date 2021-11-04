import os
import sys

sys.path.append(os.getcwd())

import numpy as np
from core.online_problem import OnlineProblem
from core.param import Param
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



#list_delta = [0.001,0.01,0.1,1,10,100]
#list_alpha = [0.001,0.01,0.1,1,10,100]

list_delta = [0.4]
list_alpha = [0.11]
m1 = len(list_delta)
m2 = len(list_alpha)
aver_values = np.zeros( (m1,m2,T) )

solver = OnlineBanditTest()
for i in range(m1):
    for j in range(m2):
        delta_i = list_delta[i]
        alpha_j = list_alpha[j]
        solver.optimize(ol_pca,X_0,delta = delta_i, alpha=alpha_j)
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

