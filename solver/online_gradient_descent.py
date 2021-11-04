import numpy as np
from .online_solver import OnlineSolver
import time
class OnlineGradientDescent(OnlineSolver):
    def __init__(self) -> None:
       self.solver_type = 'OGD' 
       self.grad_norm_his = []

    def optimize(self,problem,X_0,mu= 0):
        T = problem.time
        track_list = ['X']
        self.initial_with_problem(T,X_0,track_list)
        self.X[0] = X_0
        if mu > 0:
            self.gradient_solver_sc(problem,X_0,mu)
        else:
            self.gradient_solver(problem,X_0)
        

    def gradient_solver(self,problem,X_0):
        T = problem.time
        D = problem.param.D
        L = problem.param.L
        zeta = problem.param.zeta
        mfd = problem.mfd
        center = mfd.center
        eta = D/(L* (zeta) ** (0.5))
        for t in range(T):
            time_s = time.time()
            X_t = self.X[t]
            #suffer the loss
            value = problem.f_t(t,X_t)
            self.value_histories[t] = value
            eta_t = eta / ((t+1)**0.5)
            grad_t = problem.g_t(t,X_t)   #gradient
            self.grad_norm_his.append(np.linalg.norm(grad_t))
            X_t_plus_1 = mfd.exp(X_t, -eta_t * grad_t)
            if np.isnan(X_t_plus_1).any():
                raise ValueError
            dist_center = mfd.dist(X_t_plus_1, center)
            if dist_center >= D:            #projection
                X_t_plus_1 = mfd.exp(center,  D/dist_center * mfd.log(center,X_t_plus_1 )  )
                print(dist_center/D,t)
            self.X[t+1] =  X_t_plus_1
            time_e = time.time()
            self.time[t] = time_e-time_s
        

    def gradient_solver_sc(self,problem,X_0,mu):
        T = problem.time
        mfd = problem.mfd
        D = problem.param.D
        center = mfd.center
        eta = 1/mu
        for t in range(T):
            time_s = time.time()
            X_t = self.X[t]
            #suffer the loss
            value = problem.f_t(t,X_t)
            self.value_histories[t] = value
            #update
            eta_t = eta / (t+1)
            grad_t = problem.g_t(t,X_t)   #gradient
            self.grad_norm_his.append(np.linalg.norm(grad_t))
            X_t_plus_1 = mfd.exp(X_t, -eta_t * grad_t)

            
            if np.isnan(X_t_plus_1).any():
                raise ValueError
            self.X[t+1] =  X_t_plus_1
            time_e = time.time()
            self.time[t] = time_e-time_s
        
