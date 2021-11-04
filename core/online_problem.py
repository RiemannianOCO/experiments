import numpy as np


class OnlineProblem:
    def __init__(self,mfd,data,time,param,loss,grad):

        if time != data.shape[0]:
            raise TypeError("data error: dimension not matched")
        self.mfd = mfd
        self.dim = int(mfd.dim)
        
        self.data = data
        self.time = time
        
        self.loss = loss
        self.grad = grad
        self.param = param

    def f_t(self,time,X):
        return self.loss(self.data[time],X)

    def g_t(self,time,X):
        return self.grad(self.data[time],X)
    
    