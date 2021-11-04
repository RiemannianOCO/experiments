import sys

from pymanopt.manifolds.grassmann import Grassmann
sys.path.append('.')

from lib.function import pca
import numpy as np
import config

n = config.n
D = config.diameter
T = config.T
block=config.block

Gr = config.mfd
center = Gr.center
bound = -np.inf
lipschitz = -np.inf

A = np.load(config.foldname + 'data_A.npy')



for _ in  range(1000):
    random_index = np.random.randint(T)
    for _ in  range(100):
        X = Gr.rand()
        if Gr.dist(X,center) < D:
            value = -pca.func(A[random_index],X)
            grad_norm = Gr.norm(X, pca.grad(A[random_index],X))
            if value > bound:
                bound = value
            if grad_norm > lipschitz:
                lipschitz = grad_norm

print(bound,lipschitz)
    
