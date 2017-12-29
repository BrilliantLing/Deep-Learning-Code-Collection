# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sklearn as skl
import numpy as np
import matlab


from sklearn.neighbors import KNeighborsRegressor as KNR

today_train_data_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\train\common\today_knn'
tomorrow_train_data_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\train\common\tomorrow_knn'
today_test_data_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\test\common\today_knn'
tomorrow_test_data_dir = r'D:\MasterDL\data_set\traffic_data\2011_nhnx_speed\new\test\common\tomorrow_knn'
target_dir = r'D:\MasterDL\trans\nhnx\knn_result'

today_train_data = matlab.read_matfile_from_dir(today_train_data_dir, 'speed',[334,72*108])
tomorrow_train_data = matlab.read_matfile_from_dir(tomorrow_train_data_dir, 'speed',[334,72*108])
today_test_data = matlab.read_matfile_from_dir(today_test_data_dir,'speed',[30,72*108])
tomorrow_test_data = matlab.read_matfile_from_dir(tomorrow_test_data_dir,'speed',[30,72*108])
reality = np.reshape(tomorrow_test_data,[1,30*72*108])
KNN = KNR(n_neighbors=5)
KNN.fit(today_train_data,today_train_data)
predictions = KNN.predict(today_test_data)
matlab.save_matrix(os.path.join(target_dir,'knn_result.mat'), predictions, 'knn')
predictions = np.reshape(predictions,[1,30*72*108])
mse = ((reality-predictions)**2).mean()
print('mse:', mse)
for i in range(0,30*72*108):
    if reality[0][i] == 0:
        reality[0][i]=1
rer = np.mean(abs(reality-predictions)/reality)
mae = np.mean(abs(reality-predictions))
print('mre:', rer)
print('mae:', mae)