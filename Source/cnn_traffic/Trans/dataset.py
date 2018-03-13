# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
import shutil
import matlab

OriginDir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\fix'
TrainTodayDir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\train\today'
TrainTomorrowDir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\train\tomorrow'
TrainLastDir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\train\last'
TrainLastlastDir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\train\lastlast'

TestTodayDir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\test\today'
TestTomorrrow_dir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\test\tomorrow'
TestLastDir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\test\last'
TestLastlastDir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\test\lastlast'
TestHistoryDir = r'D:\MasterDL\data_set\traffic_data\2011_nbdx_speed\exp\test\history'

# TempOrigin = r'D:\Test\Source'
# TempTarget = r'D:\Test\Target'

random.seed(5)

def GenerateTestNum():
    BaseList = []
    TestList = []
    for x in range(100):
        BaseList.append(random.randint(1,7))
    for x in range(50):
        First = x*7+BaseList[2*x]+14
        Second = x*7+BaseList[2*x+1]+14
        while Second == First:
            Second = x*7 + random.randint(1,7)
        TestList.append(First)
        TestList.append(Second)
    return TestList

def CopyFileOfList(SourceDir, TargetDir, FilenameList):
    for x in FilenameList:
        SourcePath = os.path.join(SourceDir, x)
        TargetPath = os.path.join(TargetDir, x)
        shutil.copyfile(SourcePath, TargetPath)

def EmptyDir(Directory):
    for File in os.listdir(Directory):
        os.remove(os.path.join(Directory, File))

def EmptyDirs(DirList):
    for Directory in DirList:
        EmptyDir(Directory)

def HistoryPaths(Days, CurrIndex, CurrDir, FileList):
    DaysList = []
    for x in range(Days):
        FilePath = os.path.join(CurrDir, FileList[CurrIndex-x-1])
        DaysList.append(FilePath)
    return DaysList

def DivideDataset(origin,
                  train_lastlast, train_last, train_today, train_tomorrow,
                  test_lastlast, test_last, test_today, test_tomorrow,
                  history_dir='', history_days=1):
    TestList = GenerateTestNum()
    Origin = os.listdir(origin)
    EmptyDirs([train_lastlast, train_last, train_today, train_tomorrow, test_lastlast, test_last, test_today, test_tomorrow])
    for x in range(len(Origin)):
        if x >= 14:
            Lastlast = os.path.join(origin, Origin[x-14])
            Last = os.path.join(origin, Origin[x-7])
            Today = os.path.join(origin, Origin[x-1])
            Tomorrow = os.path.join(origin, Origin[x])
            if x in TestList:
                shutil.copyfile(Lastlast, os.path.join(test_lastlast, Origin[x-14]))
                shutil.copyfile(Last, os.path.join(test_last, Origin[x-7]))
                shutil.copyfile(Today, os.path.join(test_today, Origin[x-1]))
                shutil.copyfile(Tomorrow, os.path.join(test_tomorrow, Origin[x]))
                HistoryPathList = HistoryPaths(history_days, x, origin, Origin)
                day = matlab.read_matfile(HistoryPathList[0], 'sudushuju')
                Sum = np.zeros((day.shape[0], day.shape[1]))
                for HistoryPath in HistoryPathList:
                    Day = matlab.read_matfile(HistoryPath, 'sudushuju')
                    Sum += Day
                Mean = Sum / history_days
                matlab.save_matrix(os.path.join(history_dir, Origin[x]), Mean, 'history_mean')
                print("Day %d is test sample" %x)
            else:
                shutil.copyfile(Lastlast, os.path.join(train_lastlast, Origin[x-14]))
                shutil.copyfile(Last, os.path.join(train_last, Origin[x-7]))
                shutil.copyfile(Today, os.path.join(train_today, Origin[x-1]))
                shutil.copyfile(Tomorrow, os.path.join(train_tomorrow, Origin[x]))
                print("Day %d is train sample" %x)

def main():
    DivideDataset(OriginDir,
                  TrainLastlastDir, TrainLastDir, TrainTodayDir, TrainTomorrowDir,
                  TestLastlastDir, TestLastDir, TestTodayDir, TestTomorrrow_dir,
                  TestHistoryDir, 5)
    # FileList = os.listdir(TempOrigin)
    # HistoryFiles = HistoryPaths(2, 2, TempOrigin, FileList)
    # print(HistoryFiles)
    # day = matlab.read_matfile(HistoryFiles[0], 'speed')
    # Sum = np.zeros((day.shape[0], day.shape[1]))
    # for HistoryPath in HistoryFiles:
    #     Day = matlab.read_matfile(HistoryPath, 'speed')
    #     print(Day)
    #     Sum += Day
    # Mean = Sum / 2
    # print(Mean)

if __name__ == '__main__':
    main()    