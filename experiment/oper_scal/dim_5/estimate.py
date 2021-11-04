import sys
sys.path.append('.')
from pymanopt.manifolds import PositiveDefinite

from lib.operation.uniformly_choose import uniformly_choose
from lib.operation.generate import generate_mat
from lib.function import oper_scal
import numpy as np
import config

n = config.n
D = config.diameter
block=config.block

SPD = PositiveDefinite(n, k=1)
center = np.eye(n)
bound = -np.inf
lipschitz = -np.inf


A = generate_mat(n , 1, block, is_test=True )

for _ in  range(10000):
    delta = np.random.rand() * D
    v = uniformly_choose(center,delta,n)
    X = SPD.exp(center,v)
    value = oper_scal.func(A[0],X)
    grad_norm = SPD.norm(X, oper_scal.grad(A[0],X))
    if value > bound:
        bound = value
    if grad_norm > lipschitz:
        lipschitz = grad_norm

print(bound,lipschitz)
    
