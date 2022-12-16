import os
import sys

import numpy as np

sys.path.append(os.getcwd())

import numpy as np
from solver.online_solver import OnlineSolver
from lib.operation.uniformly_choose import uniformly_choose
import time
class OnlineBandit(OnlineSolver):
    def __init__(self) -> None:
       self.solver_type = 'BAN' 
    
    def optimize(self,problem,Y_0,mul = 1, mu = 0,c_0 = -1):
        T = problem.time
        track_list = ['X','Y']
        self.initial_with_problem(T,Y_0,track_list)

        self.Y[0] = Y_0
        if mu > 0:
            self.bandit_solver_sc(problem,Y_0,mul,mu,c_0)
        else:
            self.bandit_solver(problem,Y_0,mul)

    def bandit_solver (self,problem,Y_0,mul):
        # problem setting
        (mfd,f,T) = (   problem.mfd,
                        problem.f_t,
                        problem.time
                        )
        #params from problem

        (D,r,C,zeta) = (problem.param.D,
                        problem.param.r,
                        problem.param.C,
                        problem.param.zeta,
        )  
        #params for the algorithm
        delta = mul * (T ** (-1/4))
        tau = delta/r
        alpha = D/( C * ((zeta) * T)**(1/2) )
        center = problem.mfd.center

        for t in range(T):

            time_s = time.time()
            Y_t = self.Y[t]
            # genrate X_t
            u = uniformly_choose(mfd, Y_t , delta)
            X_t = mfd.exp(Y_t,u)
            self.X[t] = X_t
            
            # suffer from the loss
            value = f(t,X_t)

            g_t = (value / delta) * u
            Y_t_plus_1 = mfd.exp(Y_t,- alpha * g_t)
            dist_center = mfd.dist(Y_t_plus_1, center)
            
            if dist_center >= D:            
                Y_t_plus_1 = mfd.exp(center,  (1-tau)*D/(dist_center) * mfd.log(center,Y_t_plus_1 )  )
                if (dist_center/D >= 2):
                    print('warning!',t)
            self.Y[t+1] = Y_t_plus_1
            time_e = time.time()
            self.time[t] = time_e-time_s
        


    def bandit_solver_sc (self,problem,Y_0,mul,mu,c_0):
        # problem setting
        (mfd,f,T,n) = ( problem.mfd,
                        problem.f_t,
                        problem.time,
                        problem.dim
                        )
        #params from problem

        (D,r,kappa,C) = (  problem.param.D,
                            problem.param.r,
                            problem.param.kappa,
                            problem.param.C,
                            )                   
        kappa = - min(kappa,0)

        #params for the algorithm
        delta = mul * ( n * C * (1 + np.log(T) ) / T)  ** (1/3)  
        B = n / delta + n * kappa * delta
        tau = delta / r
        alpha =  B / mu 
        if c_0 == -1:
            proceed = np.round(C * alpha / D ) + 1
        else:
            proceed = c_0
        center = problem.mfd.center

        for t in range(T):
            time_s = time.time()
            Y_t = self.Y[t]
            # genrate X_t
            u = uniformly_choose(mfd, Y_t , delta)
            X_t = mfd.exp(Y_t,u)
            self.X[t] = X_t
            
            # suffer from the loss
            value = f(t,X_t)
            self.value_histories[t] = f(t,X_t)
            
            # update
            g_t = (value / delta) * u
            alpha_t = alpha / (t+proceed)
            Y_t_plus_1 = mfd.exp(Y_t,- alpha_t * g_t)
            if (np.isnan(Y_t_plus_1)).any():
                raise ValueError (('Nan: {}'.format(t)))
            dist_center = mfd.dist(Y_t_plus_1, center)
            if dist_center >= D:            
                Y_t_plus_1 = mfd.exp(center,  (1-tau)*D/dist_center * mfd.log(center,Y_t_plus_1 )  )
                if (dist_center/D >= 2):
                    print('warning!',t)
            self.Y[t+1] = Y_t_plus_1 
            time_e = time.time()
            self.time[t] = time_e-time_s