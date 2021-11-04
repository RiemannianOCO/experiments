import os
import sys

import numpy as np

sys.path.append(os.getcwd())
from core.param import Param

# dim,rounds and blocks
n=5
T=50000
block=2

# manifold 
from pymanopt.manifolds import PositiveDefinite

SPD = PositiveDefinite(n, k=1)
SPD.center = np.eye(n)

diameter = 5
lipschitz = 2
bound = 7
curvature_below = -1
curvature_above = 0

param = Param(diameter = diameter,
              curvature_above=curvature_above,
              curvature_below=curvature_below,
              lipschitz=lipschitz,
              bound = bound
              )

np.random.seed(42)
X_0 = SPD.rand()
X_0 = SPD.exp( SPD.center , diameter * SPD.log(SPD.center,X_0)/SPD.dist(SPD.center,X_0)  )
np.random.seed()

foldname = os.path.dirname(__file__) + '/data/'
