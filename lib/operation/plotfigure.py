import numpy as np
import matplotlib.pyplot as plt

grad_plot = {
    "label": "R-OGD",
    "c": "g",
}

ban_plot = {
    "label": "R-BAN",
    "c": "#ff7f0e",
}

ban_2_plot = {
    "label": "R-2-BAN",
    "c": "#1f77b4",
}

ozo_plot = {
    "label": "R-OZO",
    "c": "#d62728",
}



def load_data(foldname):
    res = {
        
        "ban":{
            "value": np.load(foldname + 'data_bandit.npy'),
            "std":   np.load(foldname + 'std_bandit.npy'),
            "time":  np.load(foldname + 'time_bandit.npy'),
            "plot":  ban_plot
        },
        "two_ban":{
            "value": np.load(foldname + 'data_two_bandit.npy'),
            "std":   np.load(foldname + 'std_two_bandit.npy'),
            "time":  np.load(foldname + 'time_two_bandit.npy'),
            "plot":  ban_2_plot
        },
    
        "ozo":{
            "value": np.load(foldname + 'data_ozo.npy'),
            "std":   np.load(foldname + 'std_ozo.npy'),
            "time":  np.load(foldname + 'time_ozo.npy'),
            "plot":  ozo_plot
        },"grad":{
            "value": np.load(foldname + 'data_gradient.npy'),
            "time":  np.load(foldname + 'time_gradient.npy'),
            "plot":  grad_plot
        }
    }

    offline = np.load(foldname + 'data_offline.npy')
    grid =  np.load(foldname + 'list_T.npy')

    return res,offline,grid

def plot_reg(res:dict,grid,std_interval):
    for alg in res.values():
        if "std" in alg:
            plt.errorbar(grid,alg["regret"],alg["std"][grid],
                elinewidth = 1,
                capsize= 2,
                errorevery=(0,std_interval),
                #barsabove=True,
                linewidth=3,
                **alg["plot"]
            )
        else:
            plt.errorbar(grid,alg["regret"],yerr=None,
                elinewidth = 1,
                capsize= 2,
                errorevery=(0,std_interval),
                #barsabove=True,
                linewidth=3,
                **alg["plot"]
            )

    plt.legend(prop={'size':16})
    plt.xlabel('Learning rounds t',fontdict={'size':18})
    plt.ylabel('E[Reg(t)] / t',fontdict={'size':18})
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)

def plot_scaled_reg(res:dict,grid,std_interval,std_start):
    for alg in res.values():
        if "std" in alg:
            plt.errorbar(grid[std_start:],alg["regret"][std_start:],alg["std"][grid][std_start:],
                elinewidth = 2,
                capsize= 2,
                errorevery=(0,std_interval),
                barsabove=True,
                linewidth=1,
                **alg["plot"]
            )
        else:
            plt.errorbar(grid[std_start:],alg["regret"][std_start:],yerr=None,
                elinewidth = 2,
                capsize= 2,
                errorevery=(0,std_interval),
                barsabove=True,
                linewidth=1,
                **alg["plot"]
            )
    plt.legend(prop={'size':16})
    plt.xlabel('Learning rounds t (scaled)',fontdict={'size':18})
    plt.ylabel('E[Reg(t)] / t',fontdict={'size':18})
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)

def plot_time(res,grid):
    for alg in res.values():
        plt.plot(alg["time"][grid],alg["regret"],linewidth=3,**alg["plot"])

    plt.legend(prop={'size':16})
    plt.xlabel('Running times',fontdict={'size':18})
    plt.ylabel('E[Reg(t)] / t',fontdict={'size':18})
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.gcf().set_facecolor(np.ones(3))
    plt.grid(True)
