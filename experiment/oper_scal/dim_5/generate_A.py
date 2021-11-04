import sys
sys.path.append('.')

from lib.operation.generate import generate_mat
import config
n=config.n
T=config.T
block=config.block
filename = config.foldname + 'data_A.npy'

A = generate_mat(n , T, block, filename)