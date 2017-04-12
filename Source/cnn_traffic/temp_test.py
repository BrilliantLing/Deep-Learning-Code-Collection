import numpy as np

mat = np.array([[1,2,3],[4,5,6],[7,8,9]])
mat_max = mat.max()
mat_min = mat.min()
mat = (mat-mat_min)/(mat_max - mat_min)
print(mat)