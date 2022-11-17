import sys
sys.path.append('.')

from lib.operation.calculate_regret import cal_reg
from lib.operation.plotfigure import *
import matplotlib.pyplot as plt
import numpy as np
import config

foldname = config.foldname
res,offline,grid= load_data(foldname)
cal_reg(res,offline,grid )
plot_reg (res,grid,std_interval= 1)
plt.show()
plot_scaled_reg (res,grid,std_interval= 1,std_start=-60)
plt.show()
