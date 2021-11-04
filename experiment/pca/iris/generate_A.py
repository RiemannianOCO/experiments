from matplotlib.pyplot import axis
from sklearn import datasets
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np
import config

foldname = config.foldname

import pandas as pd
data, target = datasets.fetch_openml('iris', version=1, return_X_y=True, as_frame=False)
ss = StandardScaler()
A = np.copy( data )
nA = ss.fit_transform(A)

filename = config.foldname + 'data_A.npy'
np.save( filename , nA.reshape(150,1,4) )
