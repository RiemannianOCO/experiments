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



list_mul = [0.001,0.01,0.1,1,2,3]
m = len(list_mul)
aver_values = np.zeros( (m,T) )
i = 0
solver = OnlineBandit()
for mul in list_mul:
    solver.optimize(ol_pca,X_0,mul)
    solver.calculate_aver_value()
    aver_values[i] = solver.aver_value_histories
    i += 1
    print('bandit solver completed',mul)
import matplotlib.pyplot as plt

for i in range(m):
    plt.plot( aver_values[i] ,label = list_mul[i])
plt.legend()
plt.show()
