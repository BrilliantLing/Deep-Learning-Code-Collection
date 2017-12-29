# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sklearn as skl
import numpy as np
import matlab
import os

from sklearn.neighbors import KNeighborsRegressor as KNR

today_train_data_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\knn\train\today'
tomorrow_train_data_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\knn\train\tomorrow'
today_test_data_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\knn\test\today'
tomorrow_test_data_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\knn\test\tomorrow'
target_dir = r'D:\MasterDL\trans\nbdx\knn'

today_train_data = matlab.read_matfile_from_dir(today_train_data_dir, 'speed',[334,43*108])
tomorrow_train_data = matlab.read_matfile_from_dir(tomorrow_train_data_dir, 'speed',[334,43*108])
today_test_data = matlab.read_matfile_from_dir(today_test_data_dir,'speed',[30,43*108])
tomorrow_test_data = matlab.read_matfile_from_dir(tomorrow_test_data_dir,'speed',[30,43*108])
reality = np.reshape(tomorrow_test_data,[1,30*43*108])
KNN = KNR(n_neighbors=5)
KNN.fit(today_train_data,today_train_data)
predictions = KNN.predict(today_test_data)
matlab.save_matrix(os.path.join(target_dir,'knn_reslut.mat'),predictions, 'knn_result')
predictions = np.reshape(predictions,[1,30*43*108])
mse = ((reality-predictions)**2).mean()
print('mse:', mse)
for i in range(0,30*43*108):
    if reality[0][i] == 0:
        reality[0][i]=1
rer = np.mean(abs(reality-predictions)/reality)
mae = np.mean(abs(reality-predictions))
print('mre', rer)
print('mae', mae)