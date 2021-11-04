from os import error
import sys
sys.path.append('.')

import numpy as np
from .online_solver import OnlineSolver
from lib.operation.uniformly_choose import uniformly_choose
import time
class OnlineBanditTest(OnlineSolver):
    def __init__(self) -> None:
       self.solver_type = 'BAN' 
    
    def optimize(self,problem,Y_0, delta, alpha, mu = 0):
        T = problem.time
        track_list = ['X','Y']
        self.initial_with_problem(T,Y_0,track_list)

        self.Y[0] = Y_0
        if mu > 0:
            self.bandit_solver_sc(problem,Y_0, delta, alpha,mu)
        else:
            self.bandit_solver(problem,Y_0,delta, alpha)

    def bandit_solver (self,problem,Y_0,delta,alpha):
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
        
        tau = delta/r
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
            # print(value)
            # update

            g_t = (value / delta) * u
            Y_t_plus_1 = mfd.exp(Y_t,- alpha * g_t)
            dist_center = mfd.dist(Y_t_plus_1, center)
            
            if dist_center >= D:            #projection
                Y_t_plus_1 = mfd.exp(center,  (1-tau)*D/dist_center * mfd.log(center,Y_t_plus_1 )  )
            self.Y[t+1] = Y_t_plus_1 
            time_e = time.time()
            self.time[t] = time_e-time_s
        


    def bandit_solver_sc (self,problem,Y_0,mul,mu):
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
        delta = mul * ( (1 + np.log(T) ) / T)  ** (1/3)  
        B = n / delta + n * kappa * delta
        tau = delta / r
        alpha = B / mu
        center = problem.mfd.center
        proceed = np.round(alpha / 0.1 ) + 1
        #proceed = 1
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
            
            #projection
            #if mfd.validate_point(Y_t_plus_1) == False:
            #    raise ValueError
            if (np.isnan(Y_t_plus_1)).any():
                raise ValueError (('Nan: {}'.format(t)))
            dist_center = mfd.dist(Y_t_plus_1, center)
            if dist_center >= D:            
                Y_t_plus_1 = mfd.exp(center,  (1-tau)*D/dist_center * mfd.log(center,Y_t_plus_1 )  )
            self.Y[t+1] = Y_t_plus_1 
            time_e = time.time()
            self.time[t] = time_e-time_s