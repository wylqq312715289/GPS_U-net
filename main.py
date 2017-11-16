#-*- coding:utf-8 -*-
import numpy as np
import os, cv2
from sklearn.cross_validation import train_test_split

from modules import myUnet, dataProcess
from config import config
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 将矩阵保存成图片
def save_imgs(test_y,files_name):
    print("array to image")
    assert test_y.shape[0]==len(files_name),"len(test_files_name) != len(test_y)"
    for i in range(test_y.shape[0]):
        img = test_y[i]
        img = img * 255
        cv2.imwrite(config.test_pred_label_path+"/%s_pred_label.png"%(files_name[i]),img)

# 数据增广,变换
def data_argmentation(imgs_train, imgs_label, expand_scale=20):
	train_x, vali_x, train_y, vali_y  = train_test_split(
		imgs_train, 
		imgs_label, 
		test_size=0.2, 
		random_state=42,
	)
	arg_train_x = []; arg_train_y = [];
	print("begin to expand training data set")
	for img_id in range(len(train_x)): 
		for i in range(expand_scale): # 整个训练集增广倍数
			new_sample_x, new_sample_y = img_random_transfer(train_x[img_id],train_y[img_id])
			arg_train_x.append(new_sample_x)
			arg_train_y.append(new_sample_y.reshape(config.img_rows,config.img_cols,1))
	print("end to expand training data set")
	return np.array(arg_train_x), vali_x, np.array(arg_train_y), vali_y 

# 初始化原始数据
def init_data():
	mydata = dataProcess(config.img_rows,config.img_cols,img_type = "png")
	mydata.create_train_data()
	mydata.create_test_data()

def main():
	# init_data() # 原始训练与测试数据没变化时,只需执行一次
	mydata = dataProcess(config.img_rows, config.img_cols, img_type = "png")
	imgs_train, imgs_label = mydata.load_train_data()
	imgs_test, files_name = mydata.load_test_data()

	train_x, vali_x, train_y, vali_y  = data_argmentation(imgs_train, imgs_label)
	print(train_x.shape,train_y.shape,vali_x.shape,vali_y.shape)
	myunet = myUnet()
	myunet.train( train_x, train_y, vali_x, vali_y )
	test_y = myunet.predict(imgs_test)
	save_imgs(test_y,files_name)

if __name__ == '__main__':
	main()





