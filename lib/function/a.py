import numpy as np
import scipy.stats

x = np.array(  [[1,2],[3,4]] )
y = np.array(  [[1,2],[3,4]] )
KL = scipy.stats.entropy(x, y) 
print(KL)