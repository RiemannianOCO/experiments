import sys
sys.path.append('.')
import os
from lib.operation.calculate_regret import cal_reg
from lib.operation.plotfigure import *
import matplotlib.pyplot as plt
import numpy as np
from data.regular import config
'''
res,offline,grid= load_data(config.foldname)
cal_reg(res,offline,grid )
plot_reg (res,grid,std_interval= 5)
plt.show()
'''
lst = ["regular","inj","diam","inj_bd","diam_bd"]
for foldname in lst:
    print(foldname)
    foldname =  os.path.dirname(__file__) + '/data/' +foldname +'/'
    res,offline,grid= load_data(foldname)
    cal_reg(res,offline,grid )
    plot_reg (res,grid,std_interval= 5)
    plt.show()
