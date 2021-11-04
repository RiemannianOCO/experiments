import sys
sys.path.append('.')

import numpy as np
from .online_solver import OnlineSolver
from lib.operation.uniformly_choose import uniformly_choose
import time
class OnlineBandit(OnlineSolver):
    def __init__(self) -> None:
       self.solver_type = 'BAN' 
    
    def optimize(self,problem,Y_0):
        T = problem.time
        track_list = ['X','Y']
        self.initial_with_problem(T,Y_0,track_list)

        self.Y[0] = Y_0
        self.bandit_solver(problem,Y_0)

    def bandit_solver (self,problem,Y_0):
        #params from problem
        (T,n,D,r,L,C,kappa,zeta,mfd,f) =    (  problem.time,
                                                problem.dim,
                                                problem.D,
                                                problem.r,
                                                problem.L,
                                                problem.C,
                                                problem.kappa,
                                                problem.zeta,
                                                problem.mfd,
                                                problem.f
                                            )                     

        #params for the algorithm
        T_shadow = T
        B = n * kappa
        big_delta = B * C * D * zeta ** (1/2) + 3 * L + 2*C /r
        small_delta = (T_shadow ** (-1/4))
        # small_delta = 1.7
        tau = small_delta/r
        alpha = D/( C * (zeta * T_shadow)**(1/2) )
        # alpha = 0.008
        center = problem.mfd.center

        for t in range(T):
            time_s = time.time()
            Y_t = self.Y[t]
            # genrate X_t
            u = uniformly_choose(mfd, Y_t , small_delta, n) # something ugly but work
            X_t = mfd.exp(Y_t,u)
            self.X[t] = X_t
            
            # suffer from the loss
            value = f(t,X_t)
            self.value_histories[t] = f(t,X_t)

            # update
            g_t = (value / small_delta) * u

            Y_t_plus_1 = mfd.exp(Y_t,- alpha * g_t)
            #dist_center = mfd.dist(Y_t_plus_1, center)
            #if dist_center >= D:            #projection
            #    Y_t_plus_1 = mfd.exp(center,  (1-tau)*D/dist_center * mfd.log(center,Y_t_plus_1 )  )
            #self.Y[t+1] = Y_t_plus_1 
            time_e = time.time()
            self.time[t] = time_e-time_s
        


