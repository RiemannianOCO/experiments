import os
import sys

sys.path.append(os.getcwd())

from lib.operation.calculate_regret import CalcuRegret
from lib.operation.plotfigure import gridplot
import matplotlib.pyplot as plt
import numpy as np
import config

foldname = config.foldname

gradient_c_aver_value = np.load(foldname + 'data_gradient.npy')
bandit_aver_value = np.load(foldname + 'data_bandit.npy')
bandit_2_aver_value = np.load(foldname + 'data_two_bandit.npy')
offline_value = np.load(foldname + 'data_offline.npy')
list_T = np.load(foldname + 'list_T.npy')
#offline_value = np.zeros(config.T)
#list_T = range(config.T)

regret_gradient_c=CalcuRegret(gradient_c_aver_value,offline_value,list_T)
regret_bandit = CalcuRegret(bandit_aver_value,offline_value,list_T)
regret_2_bandit=CalcuRegret(bandit_2_aver_value,offline_value,list_T)


gridplot(regret_2_bandit,list_T,label='R-2-BAN')
gridplot(regret_bandit,list_T,label='R-BAN')
gridplot(regret_gradient_c,list_T,label='R-OGD')
plt.legend(prop={'size':16})
plt.xlabel('Learning Rounds',fontdict={'size':18})
plt.ylabel('E[Reg(t)] / t',fontdict={'size':18})
plt.xticks(size=14)
plt.yticks(size=14)
plt.gcf().set_facecolor(np.ones(3))
plt.grid()
plt.show()



'''
grid_loglog_plot(regret_2_bandit,list_T,label='2-BAN')
grid_loglog_plot(regret_bandit,list_T,label='BAN')
grid_loglog_plot(regret_gradient_c,list_T,label='OGD-C')
plt.legend()
plt.show()
'''