import sys
import os
sys.path.append(os.getcwd())
import config
import numpy as np


n = config.n
T = config.T
block = config.block
Hn = config.mfd
center = Hn.center
np.random.seed(42)
A = np.zeros((T,block,n))
for i in range(T):
    for j in range(block):
        A[i,j] = Hn.randn()
        #A[i,j] =  Hn.exp( center, Hn.log( center , vec )/ ( Hn.dist( center , vec ) ) )

filename = config.foldname + 'data_A.npy'
np.save( filename , A )