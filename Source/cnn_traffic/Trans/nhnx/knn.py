# -*- coding: utf-8 -*-  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sklearn as skl
import numpy as np
import matlab


from sklearn.neighbors import KNeighborsRegressor as KNR

today_train_data_dir = 'D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\train\\today\\'
tomorrow_train_data_dir = 'D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\train\\tomorrow\\'
today_test_data_dir = 'D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\test\\today\\'
tomorrow_test_data_dir = 'D:\\MasterDL\\data_set\\traffic_data\\2011_yabx_speed\\test\\tomorrow\\'

today_train_data = matlab.read_matfile_from_dir(today_train_data_dir, 'speed',[319,35*288])
tomorrow_train_data = matlab.read_matfile_from_dir(tomorrow_train_data_dir, 'speed',[319,35*288])
today_test_data = matlab.read_matfile_from_dir(today_test_data_dir,'speed',[38,35*288])
tomorrow_test_data = matlab.read_matfile_from_dir(tomorrow_test_data_dir,'speed',[38,35*288])
reality = np.reshape(tomorrow_test_data,[1,38*35*288])
KNN = KNR(n_neighbors=5)
KNN.fit(today_train_data,today_train_data)
predictions = KNN.predict(today_test_data)
predictions = np.reshape(predictions,[1,38*35*288])
mse = ((reality-predictions)**2).mean()
print(mse)
for i in range(0,38*35*288):
    if reality[0][i] == 0:
        reality[0][i]=1
rer = np.mean(abs(reality-predictions)/reality)
print(rer)