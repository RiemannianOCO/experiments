import numpy as np
from .online_solver import OnlineSolver
from lib.operation.uniformly_choose import uniformly_choose
import time
class OnlineTwoPointBandit(OnlineSolver):
    def __init__(self) -> None:
       self.solver_type = 'BAN' 
    
    def optimize(self,problem,Y_0,mul = 1,mu=0, setoff = 1):
        np.random.seed()
        T = problem.time
        track_list = ['X1','X2','Y']
        self.initial_with_problem(T,Y_0,track_list)

        self.Y[0] = Y_0
        if mu > 0:
            self.bandit_solver_sc(problem,Y_0,mul,mu,setoff)
        else:
            self.bandit_solver(problem,Y_0,mul)
        

    def bandit_solver (self,problem,Y_0,mul):
                # problem setting
        (mfd,f,T) = (   problem.mfd,
                        problem.f_t,
                        problem.time
                        )
        #params from problem

        (D,r,L,zeta) = (problem.param.D,
                        problem.param.r,
                        problem.param.L,
                        problem.param.zeta
        )  
        
        #params for the algorithm
        delta = mul * T ** (-1/2)
        tau = delta/r
        alpha =  D/( delta * L * (zeta* T)**(1/2) )
        center = problem.mfd.center
        #print(alpha)
        for t in range(T):
            time_s = time.time()
            Y_t = self.Y[t]
            # genrate X_t

            u = uniformly_choose(mfd, Y_t , delta) 
            X_t_1 = mfd.exp(Y_t,u)
            self.X1[t] = X_t_1
            X_t_2 = mfd.exp(Y_t,-u)
            self.X2[t] = X_t_2
            
            # suffer from the loss
            value_1 = f(t,X_t_1)
            value_2 = f(t,X_t_2)
            value = (1/2) * (value_1 + value_2)
            self.value_histories[t] = value
            #print(value)
            
            # update
            g_t_1 = (value_1 / delta) * u
            g_t_2 = (value_2 / delta) * (-u) 
            g_t = (1/2) * (g_t_1 + g_t_2)
            #print(mfd.norm(Y_t,- alpha * g_t))
            Y_t_plus_1 = mfd.exp(Y_t,- alpha * g_t)
            #input()
            dist_center = mfd.dist(Y_t_plus_1, center)
            if dist_center >= D:            #projection
                Y_t_plus_1 = mfd.exp(center,  (1-tau)*D/dist_center * mfd.log(center,Y_t_plus_1 )  )
            self.Y[t+1] = Y_t_plus_1 
            time_e = time.time()
            self.time[t] = time_e-time_s
            

    def bandit_solver_sc (self,problem,Y_0, mul,mu,setoff):
        # problem setting
        (mfd,f,T,n) = ( problem.mfd,
                        problem.f_t,
                        problem.time,
                        problem.dim
                        )
        #params from problem

        (D,r,kappa,L) = (  problem.param.D,
                            problem.param.r,
                            problem.param.kappa,
                            problem.param.L
                            )                   
        kappa = - min(kappa,0)
                                    
        
        #params for the algorithm
        delta = mul * (1 + np.log(T)) / T
        B = n/delta + n * kappa * delta
        #B = n/delta 
        tau = delta/r
        alpha = B / mu
        center = problem.mfd.center
        for t in range(T):
            time_s = time.time()
            Y_t = self.Y[t]
            # genrate X_t
            u = uniformly_choose(mfd, Y_t , delta) # something ugly but work
            X_t_1 = mfd.exp(Y_t,u)
            self.X1[t] = X_t_1
            X_t_2 = mfd.exp(Y_t,-u)
            self.X2[t] = X_t_2
            
            # suffer from the loss
            value_1 = f(t,X_t_1)
            value_2 = f(t,X_t_2)
            value = (1/2) * (value_1 + value_2)
            self.value_histories[t] = value

            # update
            g_t_1 = (value_1 / delta) * u
            g_t_2 = (value_2 / delta) * (-u) 
            g_t = (1/2) * (g_t_1 + g_t_2)
            alpha_t = alpha / (t+setoff)

            Y_t_plus_1 = mfd.exp(Y_t,- alpha_t * g_t)
            dist_center = mfd.dist(Y_t_plus_1, center)
            if dist_center >= D:            #projection
                Y_t_plus_1 = mfd.exp(center,  (1-tau)*D/dist_center * mfd.log(center,Y_t_plus_1 )  )
            self.Y[t+1] = Y_t_plus_1 
            time_e = time.time()
            self.time[t] = time_e-time_s
            

