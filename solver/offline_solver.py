from pymanopt import Problem,function
from pymanopt.optimizers import *
import numpy as np
class OfflineSolver():
    def __init__(self, solver, mingrad = 10e-6, max_time = 10000) -> None:
        self.solver = solver(min_gradient_norm = mingrad,max_time = max_time)
        self.solver._verbosity = 2
    
    def optimize(self,ol_problem,X_0,list_T):
        self.list_T = list_T
        length = len(list_T)
        self.offline_histories = np.zeros( length )
        return self.offline_solver(ol_problem,X_0,list_T)
        
    def offline_solver(self,problem,X_0,list_T):
        length = len(list_T)
        for i in range(length):
            t = self.list_T[i]
            print('offline round:',t)
            @function.numpy(problem.mfd)
            def func(X):
                return problem.sum_f(t,X)
            @function.numpy(problem.mfd)
            def grad(X):
                return problem.sum_g(t,X)
            off_problem = Problem(manifold = problem.mfd, cost=func, riemannian_gradient=grad)
            Xopt = self.solver.run(off_problem,initial_point =X_0).point
            dist_center = problem.mfd.dist(Xopt, X_0)
            #print(dist_center)
            print('value',func(Xopt))
            self.offline_histories[i]= func(Xopt)
        return Xopt