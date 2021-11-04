import numpy as np
from hyperbolic import HyperbolicSpace
n = 10
Hn = HyperbolicSpace( n )


# test rand()
X = Hn.rand()
if ( np.abs(Hn.mdot(X,X) + 1) < 1e-13):
    print('test rand(): completed')
else:
    raise ValueError('test rand(): failed')

# test log
X = Hn.rand()
Y = Hn.rand()
v = Hn.log(X,Y)
if  (Y - Hn.exp(X,v) < 1e-13).all():
    print('test log(): completed')
else:
    raise ValueError('test log(): failed')



