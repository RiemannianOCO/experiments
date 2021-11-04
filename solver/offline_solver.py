from pymanopt import Problem
from pymanopt.solvers import *
import numpy as np
class OfflineSolver():
    def __init__(self, solver, mingrad = 10e-6, maxtime = 10000) -> None:
        self.solver = solver(mingradnorm = mingrad,maxtime = 10000)
    
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
            func = lambda X: problem.sum_f(t,X)
            grad = lambda X: problem.sum_g(t,X)
            off_problem = Problem(manifold = problem.mfd, cost=func, grad=grad)
            Xopt = self.solver.solve(off_problem,x=X_0)
            dist_center = problem.mfd.dist(Xopt, X_0)
            print(dist_center)
            print('value',func(Xopt))
            self.offline_histories[i]= func(Xopt)
        return Xopt