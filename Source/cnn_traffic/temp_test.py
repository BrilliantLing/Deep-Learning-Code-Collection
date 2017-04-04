import numpy as np
array=np.arange(16).reshape(4,4)
array[:,0] = array[:,2]+ array[:,3]

print(array)