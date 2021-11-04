import sys
sys.path.append('.')

from lib.operation.calculate_regret import CalcuRegret
from lib.operation.plotfigure import gridplot,grid_loglog_plot,grid_time_plot
import matplotlib.pyplot as plt
import numpy as np
from experiment.fre_mean.hyp_100 import config

fold_off = config.foldname 
offline_value = np.load(fold_off + 'data_offline.npy')
list_T = np.load(fold_off + 'list_T.npy')

fold_ol = config.fold_strong

gradient_sc_aver_value = np.load(fold_ol + 'data_gradient.npy')
ozo_aver_value = np.load(fold_ol + 'data_ozo.npy')
bandit_aver_value = np.load(fold_ol + 'data_bandit.npy')
two_bandit_aver_value = np.load(fold_ol + 'data_two_bandit.npy')


regret_gradient_sc=CalcuRegret(gradient_sc_aver_value,offline_value,list_T)
regret_ozo=CalcuRegret(ozo_aver_value,offline_value,list_T)
regret_bandit = CalcuRegret(bandit_aver_value,offline_value,list_T)
regret_two_bandit = CalcuRegret(two_bandit_aver_value,offline_value,list_T)


gradient_time = np.load(fold_ol + 'time_gradient.npy')
bandit_time = np.load(fold_ol + 'time_bandit.npy')
bandit_2_time = np.load(fold_ol + 'time_two_bandit.npy')
ozo_time = np.load(fold_ol + 'time_ozo.npy')



gridplot(regret_two_bandit,list_T,label='R-2-BAN')
gridplot(regret_bandit,list_T,label='R-BAN')
gridplot(regret_gradient_sc,list_T,label='R-OGD')
gridplot(regret_ozo,list_T,label='R-OZO')
plt.legend(prop={'size':16})
plt.xlabel('Learning rounds t',fontdict={'size':18})
plt.ylabel('E[Reg(t)] / t',fontdict={'size':18})
plt.xticks(size=14)
plt.yticks(size=14)
plt.gcf().set_facecolor(np.ones(3))
plt.grid(True)
plt.show()

grid_time_plot(bandit_2_time,regret_two_bandit,list_T,label='R-2-BAN')
grid_time_plot(bandit_time,regret_bandit,list_T,label='R-BAN')
grid_time_plot(gradient_time,regret_gradient_sc,list_T,label='R-OGD')
grid_time_plot(ozo_time,regret_ozo,list_T,label='R-OZO')
plt.legend(prop={'size':16})
plt.xlabel('Ruuning times',fontdict={'size':18})
plt.ylabel('E[Reg(t)] / t',fontdict={'size':18})
plt.xticks(size=14)
plt.yticks(size=14)
plt.gcf().set_facecolor(np.ones(3))
plt.grid(True)
plt.show()
