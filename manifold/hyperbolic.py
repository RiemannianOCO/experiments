import numpy as np
import scipy.linalg as la
from pymanopt.manifolds.manifold import Manifold


class HyperbolicSpace(Manifold):

    def mdot(self,X,Y):
        X_m = np.copy(X)
        X_m[-1] = -X_m[-1]
        return np.dot(Y,X_m)

    def __init__(self,n):
        self._n = n # embed in the R^{n}
        self._name = ("Hyperbolic manifold of {}-vectors").format(self._n)
        self._point_layout = 1

    def __str__(self):
        return self._name
    
    @property
    def dim(self):
        return self._n - 1

    def dist(self, X , Y):
        return  np.arccosh( - self.mdot(X,Y) )
    
    def inner_product(self,X,U,V):
        return self.mdot(U,V)
    
    def norm(self, X ,U ):
        return np.sqrt( self.mdot(U,U) )

    def exp(self,X,U):
        # validate
        #if self.validate_vector(X,U) == False:
        #    raise ValueError("exp error: U is not on T_M X")
        norm_U = self.norm(None , U)
        return X * np.cosh(norm_U) + U * np.sinh(norm_U) / norm_U
    
    def retr(self, X, U):
        return self.exp(X, U)

    def log(self,X,Y):
        mdot= self.mdot(X , Y)
        dist = self.dist(X , Y)
        if len(Y.shape) == 1:
            vec = Y + mdot * X
            unit_vec =  ( mdot**2 -1 ) ** -(0.5) * vec
            return dist * unit_vec
        else:
            vec = Y + np.outer(mdot,X)
            unit_vec = np.diag( ( mdot**2 -1 ) ** -(0.5) ) @ vec
            return np.diag(dist) @ unit_vec

    def projection(self,X , U):
        mdot= self.mdot(X ,U)
        if len(U.shape) == 1:
            return  U + mdot * X
        else:
            return U + np.outer(mdot,X)

    def random_point(self):
        X = np.zeros(self._n)
        X[:-1] = np.random.randn(self._n - 1) 
        X[-1] = np.sqrt( X[:-1] @ X[:-1] + 1 )
        return X

    def randn(self):
        X = np.zeros(self._n)
        X[:-1] = np.random.randn(self._n - 1) / np.sqrt(self._n - 1)
        X[-1] = np.sqrt( X[:-1] @ X[:-1] + 1 )
        return X

    
    def random_tangent_vector(self, X):
        H = np.random.randn(self._n)
        P = self.projection(X, H)
        return self._normalize(P)

    def validate_point(self,X):
        if np.abs( self.mdot(X,X) + 1 ) >= 1e-10:
            return False
        return True


    def validate_vector(self,X,U):
        if self.mdot(X,U) >= 1e-10:
            return False
        return True

    def _normalize(self,X):
        return X / self.norm(None , X)

    def zero_vector(self, point):
        return np.zeros(self._n)

    retraction = exp