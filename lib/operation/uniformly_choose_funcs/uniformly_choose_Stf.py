import numpy as np
from scipy import linalg as la


def uniformly_choose_Stf(manifold, X, delta):

    return manifold.randvec(X) * delta

