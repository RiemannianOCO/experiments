from os import error
import sys
sys.path.append('.')

import numpy as np
from .online_solver import OnlineSolver
from lib.operation.uniformly_choose import uniformly_choose
import time
class OnlineZeroth(OnlineSolver):
    def __init__(self) -> None:
       self.solver_type = 'OZO' 
    
    def optimize(self,problem,X_0,sigma,L,V):
        T = problem.time
        track_list = ['X']
        self.initial_with_problem(T,X_0,track_list)

        self.X[0] = X_0
        self.zeroth_solver(problem,X_0,sigma,L,V)

    def zeroth_solver (self,problem,X_0,sigma,L,V):
        np.random.seed()
           # problem setting
        (mfd,f,T,n) = ( problem.mfd,
                        problem.f_t,
                        problem.time,
                        problem.dim
                        )
        #params from problem

        (D,C,zeta) = (  problem.param.D,                   
                        problem.param.C,
                        problem.param.zeta,
        )    

        delta = 0.001
        eta = (4 * (delta ** 2) * n /( (L ** 2) * (n + 6) ** 3) ) ** ( 0.25 )
        theta =  0.5 * (L * eta) ** 2  * (n + 6) ** 3
        theta += 2 * L * delta * ( n + 4 ) ** 2
        theta += 2 * (delta / eta) ** 2 * n
        theta *= zeta
        theta = theta ** (0.5)

        A = ( 8 * V * L ** 2 * zeta * ( n + 4 ) + sigma * theta ) ** 2 - 8 * sigma ** 2 * theta ** 2 * L ** 2 *  zeta * ( n + 4 ) 
        B = -4 *  V * ( 8 * V * zeta * L ** 2 * ( n + 4 ) * sigma + theta * sigma ** 2 + 8 * theta ** 2 * zeta * L ** 3 * ( n + 4 ) )
        C = (2 * sigma * V + 2 * theta) ** 2 - 4 * theta ** 2
        thre = 1/ ( 2 * L ** 2 * zeta * (n + 4 ) )
        arg = [A,B,C]
        alpha1, alpha2 = np.roots(arg)
    
        if (alpha1 > 0) and (alpha1 < thre):
            alpha = alpha1
        elif (alpha2 > 0) and (alpha2 < thre):
            alpha = alpha2
        else:
            raise ValueError
        center = mfd.center
        for t in range(T):

            time_s = time.time()
            X_t = self.X[t]
            # genrate u

            u = np.random.randn( *(mfd.random_point().shape) )
            Pu = mfd.projection(X_t , u)
            Y_t = mfd.exp(X_t , eta * Pu)
            # update X_t_plus_1 
            g_t = ( (f(t,Y_t) - f(t,X_t)) / eta ) * Pu
            self.value_histories[t] = 0.5 * ( f(t,X_t) + f(t,Y_t) )
            X_t_plus_1 = mfd.exp(X_t, -alpha * g_t)
            
            #projection
            dist_center = mfd.dist(X_t_plus_1, center)
            if dist_center >= D:            
                X_t_plus_1 = mfd.exp(center,  D / dist_center * mfd.log(center,X_t_plus_1 )  )
            self.X[t+1] = X_t_plus_1             
            
            time_e = time.time()
            self.time[t] = time_e-time_s
        


    