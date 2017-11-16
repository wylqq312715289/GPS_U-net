#-*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def img_random_transfer(img,label):
	rows, cols = img.shape[:2]
	shift_style = random.randint(0,1) # 开区间
	shift_style = 2
	if random.randint(0,1)==1: # 随机,不翻转, 水平, 上下, 水平+上下
		flip_flag = random.randint(-1,1)
		img = cv2.flip(img,flip_flag)
		label = cv2.flip(label,flip_flag)
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
	print(img.shape)
	plt.subplot(121)
	plt.imshow(img)
	plt.subplot(122)
	plt.imshow(new_img)
	plt.show()

if __name__ == '__main__':
	# for i in range(100):
	# 	print(random.randint(0,4))
	img = cv2.imread('test.png')
	img_random_transfer( img, img )


