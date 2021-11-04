import sys
sys.path.append('.')

from lib.operation.calculate_regret import CalcuRegret
from lib.operation.plotfigure import gridplot,grid_loglog_plot,grid_time_plot
import matplotlib.pyplot as plt
import numpy as np
import config

foldname = config.foldname
gradient_aver_value = np.load(foldname + 'data_gradient.npy')
bandit_aver_value = np.load(foldname + 'data_bandit.npy')
bandit_2_aver_value = np.load(foldname + 'data_two_bandit.npy')
offline_value = np.load(foldname + 'data_offline.npy')
list_T = np.load(foldname + 'list_T.npy')

gradient_time = np.load(foldname + 'time_gradient.npy')
bandit_time = np.load(foldname + 'time_bandit.npy')
bandit_2_time = np.load(foldname + 'time_two_bandit.npy')


regret_gradient=CalcuRegret(gradient_aver_value,offline_value,list_T)
regret_bandit = CalcuRegret(bandit_aver_value,offline_value,list_T)
regret_2_bandit=CalcuRegret(bandit_2_aver_value,offline_value,list_T)


gridplot(regret_2_bandit,list_T,label='R-2-BAN')
gridplot(regret_bandit,list_T,label='R-BAN')
gridplot(regret_gradient,list_T,label='R-OGD')
plt.legend(prop={'size':16})
plt.xlabel('Learning rounds t',fontdict={'size':18})
plt.ylabel('E[Reg(t)] / t',fontdict={'size':18})
plt.xticks(size=14)
plt.yticks(size=14)
plt.gcf().set_facecolor(np.ones(3))
plt.grid(True)
plt.show()

grid_time_plot(bandit_2_time,regret_2_bandit,list_T,label='R-2-BAN')
grid_time_plot(bandit_time,regret_bandit,list_T,label='R-BAN')
grid_time_plot(gradient_time,regret_gradient,list_T,label='R-OGD')
plt.legend(prop={'size':16})
plt.xlabel('Ruuning times',fontdict={'size':18})
plt.ylabel('E[Reg(t)] / t',fontdict={'size':18})
plt.xticks(size=14)
plt.yticks(size=14)
plt.gcf().set_facecolor(np.ones(3))
plt.grid(True)
plt.show()