from numpy.linalg import eigh
from pymanopt.manifolds import Grassmann
import config
import numpy as np

n=config.n
T=config.T
d=config.d
block=config.block
foldname = config.foldname

Gr =config.mfd
B = np.load(foldname + 'data_A.npy')
random_index = np.random.randint(config.T)
A = B[random_index].T @ B[random_index]
v,w = eigh(A)
B = 0


Y = np.zeros((n,d))
Y[:d,:d] = np.eye(d)
Y[:,0] = w[:,-1]
#Y = Gr.rand()
D = Gr.randvec(Y)
#print(Gr.dist(Y,Gr.center))
#Y = Gr.exp(Y, 2.3 * np.random.random() * D)



D = Gr.randvec(Y)
D /= Gr.norm(Y,D)

egrad = -A @ Y
ehess = -(np.eye(n) - Y@ Y.T) @ A @ D

print( Gr.inner(Y, D, Gr.ehess2rhess(Y,egrad,ehess,D) ) )