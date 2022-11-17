import os
import sys

import numpy as np

sys.path.append(os.getcwd())
from core.param import Param

# dim,rounds and blocks
n=101
T=10000
block=10

# manifold 
from manifold.hyperbolic import HyperbolicSpace
mfd = HyperbolicSpace(n)
mfd.center = np.array([0]*(n-1)+[1])

# parameter
diameter = 5
lipschitz = 5.5
bound = 15
curvature_below = -1
curvature_above = -1

param = Param(diameter = diameter,
              curvature_above=curvature_above,
              curvature_below=curvature_below,
              lipschitz=lipschitz,
              bound = bound
              )

# initial point
np.random.seed(42)
X_0 = mfd.random_point()
X_0 = mfd.exp( mfd.center , diameter * mfd.log(mfd.center,X_0) / mfd.dist(mfd.center,X_0) )
np.random.seed()

# save file
foldname = os.path.dirname(__file__)+ '/data/'
