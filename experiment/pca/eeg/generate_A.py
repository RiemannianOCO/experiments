from matplotlib.pyplot import axis
from sklearn import datasets
import sklearn
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
foldname = os.path.dirname(__file__) + '/data/regular/'

import pandas as pd
data, target = datasets.fetch_openml('eeg-eye-state', version=1, return_X_y=True, as_frame=False)
ss = StandardScaler()
A = np.copy( data )
nA = ss.fit_transform(A)
d = []
#delete outliers
for i in range(nA.shape[0]):
    if (nA[i]>3).any() or (nA[i]<-3).any():
        d.append(i)
nA =np.delete(nA,d,axis=0)
nA = nA[:14000]
np.random.shuffle(nA)
filename = foldname + 'data_A.npy'
np.save( filename , nA.reshape(14000,1,14) )
