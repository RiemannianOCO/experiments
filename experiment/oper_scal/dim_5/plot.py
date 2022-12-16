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
plot_reg (res,grid,std_interval= 5)
plt.show()
plot_scaled_reg (res,grid,std_interval= 5,std_start=-100)
plt.xticks([30000,35000,40000,45000,50000])
plt.show()
plot_time(res,grid)
plt.ylim(top=3.3)
plt.show()
