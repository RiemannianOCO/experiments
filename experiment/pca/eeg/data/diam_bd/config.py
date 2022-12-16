import os
import sys

import numpy as np

sys.path.append(os.getcwd())
from core.param import Param


# dim,rounds and blocks
n=14
d=3
T=14000
block=1

# manifold 
from pymanopt.manifolds import Grassmann

#regular stands for
#
foldname = os.path.dirname(__file__) +'/'

mfd = Grassmann(n,d)
X = np.load(foldname+'X.npy')
mfd.center = X

curvature_above = 2 
lipschitz = 1

diameter = d **2 * np.pi/2

bound = 2
curvature_below = 0

param = Param(diameter = diameter,
              curvature_above=curvature_above,
              curvature_below=curvature_below,
              lipschitz=lipschitz,
              bound = bound
              )

np.random.seed(42)
X_0 = mfd.random_point()
X_0 = mfd.exp( mfd.center , (0.999* diameter) * mfd.log(mfd.center,X_0)/mfd.dist(mfd.center,X_0)  )
np.random.seed()



