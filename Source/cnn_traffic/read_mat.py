# -*- coding: utf-8 -*-  
import numpy as np
import scipy.io as sio
import os

#the dirctories that the mat files is stored
flow_data_dir = 'D:\\MasterDL\\data_set\\2011_flow\\'
speed_data_dir = 'D:\\MasterDL\\data_set\\2011_speed\\'
occupancy_data_dir = 'D:\\MasterDL\\data_set\\2011_occupancy\\'
test_dir = 'D:\\test'
test_dir_a = 'D:\\test\\A\\'
test_dir_b = 'D:\\test\\B\\'

#the dirctories that the numpy files or tfrecords files to be saved
target_dir = 'D:\\MasterDL\\data_set\\'

def read_matfiles(data_dir, variable_name, shape):
    mat_array = np.array([])
    for matfile in os.listdir(data_dir):
        mat = sio.loadmat(data_dir+matfile)
        mat = mat[variable_name]
        mat_array = np.append(mat_array, mat)
 
    mat_array=np.reshape(mat_array,shape)
    return mat_array

def create_numpy_record(matrix, target_dir, filename):
    np.save(matrix, target_dir+filename)

def merge_2channels(mat_a, mat_b ,shape):
    mat_array = np.array([])
    for index in range(mat_a.shape[0]):
        mat_array = np.append(mat_array,mat_a[index])
        mat_array = np.append(mat_array,mat_b[index])
    
    mat_array = np.reshape(mat_array,shape)
    return mat_array

def merge_3channels(mat_a, mat_b, mat_c, shape):
    mat_array = np.array([])
    for index in range(mat_a.shape[0]):
        mat_array = np.append(mat_array,mat_a[index])
        mat_array = np.append(mat_array,mat_b[index])
        mat_array = np.append(mat_array,mat_c[index])
    
    mat_array = np.reshape(mat_array, shape)
    return mat_array

def main():
    mat_a = read_matfiles(test_dir_a,'A',(2,3,3))
    mat_b = read_matfiles(test_dir_b,'B',(2,3,3))
    matrix = merge_2channels(mat_a,mat_b,(2,2,3,3))
    print(matrix)
    pass

if __name__ == '__main__':
    main()
