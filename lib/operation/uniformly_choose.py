from pymanopt import manifolds
from manifold.hyperbolic import HyperbolicSpace
from pymanopt.manifolds import *
from .uniformly_choose_funcs import *
from lib.operation import uniformly_choose_funcs
def uniformly_choose(manifold, X , delta):
    if isinstance(manifold, SymmetricPositiveDefinite):
        return uniformly_choose_SPD(X , delta , X.shape[0])

    elif isinstance(manifold, Stiefel):
        return uniformly_choose_Stf(manifold, X , delta)

    elif isinstance(manifold, HyperbolicSpace):
        return uniformly_choose_hyp(manifold, X ,delta)
    
    elif isinstance(manifold, Sphere):
        return uniformly_choose_Sphere(manifold, X ,delta)

    else:
        return manifold.random_tangent_vector(X) * delta
