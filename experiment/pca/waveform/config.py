import os
import sys

import numpy as np

sys.path.append(os.getcwd())
from core.param import Param

foldname = os.path.dirname(__file__) + '/data/'
# dim,rounds and blocks
n=40
d=10
T=1000
block=5

# manifold 
from pymanopt.manifolds import Grassmann
mfd = Grassmann(n,d)
X = np.load(foldname+'X_0.npy')
mfd.center = X

curvature_above = 2
diameter = np.pi / ( 2*np.sqrt(curvature_above) )
lipschitz = 10
bound = 3
curvature_below = 0

param = Param(diameter = diameter,
              curvature_above=curvature_above,
              curvature_below=curvature_below,
              lipschitz=lipschitz,
              bound = bound
              )


np.random.seed(99)
X_0 = mfd.random_point()
X_0 = mfd.exp( mfd.center , (0.999* diameter) * mfd.log(mfd.center,X_0)/mfd.dist(mfd.center,X_0)  )
# X_0 =mfd.center
np.random.seed()


print(foldname)
