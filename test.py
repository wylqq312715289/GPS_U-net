#-*- coding:utf-8 -*-
import numpy as np
import csv, sys, cv2, os, math
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def plot_label_img(file_path):
	# file_path ="./cache/test_imgs/pred_labels/0.png"
	img = cv2.imread(file_path,flags=cv2.IMREAD_UNCHANGED)
	print img.shape
	img = cv2.resize(img, (512, 512))
	print img.shape
	# img *= 255
	# print( np.min(img),np.max(img) )
	# img[img>0.5] = 1
	# img[img<=0.5] = 0
	# print( np.min(img),np.max(img) )
	print( np.sum(img) )
	cv2.imshow('image',img*256)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def plot_train_img(file_path):
	# file_path ="./cache/test_imgs/pred_labels/0.png"
	img = cv2.imread(file_path,flags=cv2.IMREAD_UNCHANGED)
	print img.shape
	img = cv2.resize(img, (512, 512))
	print img.shape
	img *= 255
	print( np.min(img),np.max(img) )
	# img[img>0.5] = 1
	# img[img<=0.5] = 0
	print( np.min(img),np.max(img) )
	print( np.sum(img) )
	cv2.imshow('image',img*256)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	file_path = "./cache/test_imgs/pred_labels/0.png"
	# file_path = "./cache/train_imgs/labels/LC81240322017118LGN00_split_000_json.png"
	plot_label_img(file_path)


