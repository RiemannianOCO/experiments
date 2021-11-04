import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from lib.function.frechet_mean_h import func , grad 
from manifold.hyperbolic import HyperbolicSpace


n = 10
Hn = HyperbolicSpace( n )
block = 10

X = Hn.rand()
dist = 0
g = np.zeros(n)
A = np.zeros( ( block , n )  )
for i in range(block):
    A[i] = Hn.rand()
    dist += Hn.dist(X,A[i])
    g += Hn.log(X,A[i])

dist /= block
test_func = func(A,X)

g /= block
test_grad = grad(A,X)


print(dist-test_func)
print(g-test_grad)