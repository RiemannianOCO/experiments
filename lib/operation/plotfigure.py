import matplotlib.pyplot as plt


def grid_loglog_plot(f,list_T,label=''):
    plt.loglog(list_T,f,label = label)

def gridplot(f,list_T,label=''):
    plt.plot(list_T,f,label = label,linewidth=4)
   

def grid_time_plot(time,regret,list_T,label=''):
    plt.plot(time[list_T],regret,label = label,linewidth=4)
    #plt.yscale('log')