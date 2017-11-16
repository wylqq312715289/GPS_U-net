#!/usr/bin/env python  
# -*- coding: UTF-8 -*-  
from datetime import datetime  
from datetime import timedelta  
import pandas as pd  
import os, copy, math, time, h5py, json, random
import numpy as np
import cv2
from sklearn import preprocessing  
from sklearn.datasets import dump_svmlight_file  
# from svmutil import svm_read_problem


# 一张图像随机变换
def img_random_transfer(img,label):
    if random.randint(0,1) == 1: 
        # 随机,不翻转, 水平, 上下, 水平+上下
        flip_flag = random.randint(-1,1)
        img = cv2.flip(img,flip_flag)
        label = cv2.flip(label,flip_flag)
    rows, cols = img.shape[:2]
    shift_style = random.randint(0,1) # 开区间
    if shift_style == 0: # 平移变换
        x_move = random.randint(-100,100)
        y_move = random.randint(-100,100)
        M = np.float32([[1,0,x_move],[0,1,y_move]]) #平移矩阵1：向x正方向平移25，向y正方向平移50
    elif shift_style == 1: # 旋转变换
        angle = random.randint(-180,180)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1) # 最后一个参数为缩放因子
    elif shift_style == 2: # 仿射变换
        pts1 = np.float32([[50,50],[200,50],[50,200]])
        pts2 = np.float32([[10,100],[200,50],[100,250]])
        M = cv2.getAffineTransform(pts1,pts2)
    new_img = cv2.warpAffine(img,M,(rows,cols)) #需要图像、变换矩阵、变换后的大小
    new_label = cv2.warpAffine(label,M,(rows,cols))
    return new_img, new_label

# 一般矩阵归一化  
def my_normalization( data_ary, axis=0 ):  
    # axis = 0 按列归一化; 1时按行归一化  
    if axis == 1:  
        data_ary = np.matrix(data_ary).T  
        ans = preprocessing.scale(data_ary)  
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1.0,1.0))  
        ans = min_max_scaler.fit_transform(ans)  
        ans = np.matrix(ans).T  
        ans = np.array(ans)  
    else:  
        ans = preprocessing.scale(data_ary)  
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1.0,1.0))  
        ans = min_max_scaler.fit_transform(ans)  
    return ans  
  
def one_hot( data_ary, one_hot_len):  
    # data_ary = array([1,2,3,5,6,7,9])  
    # one_hot_len: one_hot最长列  
    max_num = np.max(data_ary);  
    ans = np.zeros((len(data_ary),one_hot_len),dtype=np.float64)  
    for i in range(len(data_ary)):  
        ans[ i, int(data_ary[i]) ] = 1.0  
    return ans  
  
def re_onehot( data_ary ):  
    # data_ary = array([[0,0,0,1.0],[1.0,0,0,0],...])  
    ans = np.zeros((len(data_ary),),dtype=np.float64)  
    for i in range(len(data_ary)):  
        for j in range(len(data_ary[i,:])):  
            if data_ary[i,j] == 1.0:  
                ans[i] = 1.0*j;  
                break;  
    return ans  
  
# 将数据写入h5文件  
def write2H5(h5DumpFile,data):  
    # if not os.path.exists(h5DumpFile): os.makedir(h5DumpFile)  
    with h5py.File(h5DumpFile, "w") as f:  
        f.create_dataset("data", data=data, dtype=np.float64)  
  
# 将数据从h5文件导出  
def readH5(h5DumpFile):  
    feat = [];  
    with h5py.File(h5DumpFile, "r") as f:  
        feat.append(f['data'][:])  
    feat = np.concatenate(feat, 1)  
    print('readH5 Feature.shape=', feat.shape)  
    return feat.astype(np.float64)  
  
# 将dict数据保存到json  
def store_json(file_name,data):  
    with open(file_name, 'w') as json_file:  
        json_file.write(json.dumps(data,indent=4))  
  
# 将json文件中的数据读取到dict  
def load_json(file_name):  
    with open(file_name) as json_file:  
        data = json.load(json_file)  
        return data  
  
#将一个文件copy到指定目录  
def moveFileto( sourceDir, targetDir ): shutil.copy( sourceDir, targetDir )  
  
# 删除目录下的所有文件  
def removeDir(dirPath):  
    if not os.path.isdir(dirPath): return  
    files = os.listdir(dirPath)  
    try:  
        for file in files:  
            filePath = os.path.join(dirPath, file)  
            if os.path.isfile(filePath):  
                os.remove(filePath)  
            elif os.path.isdir(filePath):  
                removeDir(filePath)  
        os.rmdir(dirPath)  
    except Exception, e: print e