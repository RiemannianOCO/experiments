import sys
import os
import numpy as np
from lib.function.std_func import sigma, zeta

class Param:
    def __init__(self, diameter = None, curvature_above  = None, curvature_below  = None, lipschitz  = None,  bound  = None) -> None:
        #validation
        Val_err = 'param missing: {}'
        if diameter == None:
            raise ValueError(Val_err.format('diameter'))
        if curvature_below == None:
            raise ValueError(Val_err.format('curvature_below'))
        if curvature_above == None:
            raise ValueError(Val_err.format('curvature_above'))
        if (lipschitz==None) and (bound == None):
            raise ValueError(Val_err.format('lipschitz or bound'))

        self.D = diameter
        self.L = lipschitz 
        self.K = curvature_above
        self.kappa = curvature_below
        self.C = bound
        self.r = diameter

    @property
    def zeta(self):
        return zeta(self.kappa,self.D)