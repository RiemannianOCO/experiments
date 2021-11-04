import os
import sys

import numpy as np

sys.path.append(os.getcwd())
from core.param import Param


# dim,rounds and blocks
n=4
d=2
T=150
block=1

# manifold 
from pymanopt.manifolds import Grassmann
np.random.seed(42)
mfd = Grassmann(n,d)
X = np.zeros((n,d))
X[:d] = np.eye(d)
mfd.center = X

curvature_above = 0.5
diameter = np.pi / ( 2*np.sqrt(curvature_above) )
lipschitz = 6
bound = 6
curvature_below = 0

param = Param(diameter = diameter,
              curvature_above=curvature_above,
              curvature_below=curvature_below,
              lipschitz=lipschitz,
              bound = bound
              )


np.random.seed(42)
X_0 = mfd.rand()
X_0 = mfd.exp( mfd.center , (0.99* diameter) * mfd.log(mfd.center,X_0)/mfd.dist(mfd.center,X_0)  )
X_0 =mfd.center
np.random.seed()

foldname = os.path.dirname(__file__) + '/data/'
print(foldname)
