import numpy


import numpy as np
class OfflineProblem:
    def __init__(self,mfd, data, time,loss = None,grad = None,sum_loss = None,sum_grad = None) -> None:
        if sum_loss and sum_grad:
            self.flag = True
            self.sum_loss = sum_loss
            self.sum_grad = sum_grad
        elif loss and grad:
            self.flag = False
            self.f = loss
            self.g = grad
        else:
            raise  ValueError('No function in offline problem')

        self.mfd = mfd
        self.data = data
        self.T = time


    def sum_f(self,time,X):
        if self.flag:
            return self.sum_loss(self.data[:(time+1)],X)
        else:
            value = 0
            for i in range(time+1):
                value = value + self.f(self.data[i],X)
            return (1/(time+1)) * value
    
    def sum_g(self,time,X):
        if self.flag:
            return self.sum_grad(self.data[:(time+1)],X)
        else:
            ans = self.g(self.data[0],X)
            for i in range(time):
                ans = ans + self.g(self.data[i+1],X)
            return (1/(time+1)) * ans

    def random_g(self,time,X):
        i = np.random.randint(time+1)
        grad = self.g(i,X)
        return grad