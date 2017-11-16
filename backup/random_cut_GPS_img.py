#-*- coding:utf-8 -*-
import numpy as np
import csv, sys, cv2, os, math
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import tifffile as tiff
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm

file_idx = [4,5,6] #组合三个波段的图像为一张图像

# 随机裁剪512*512的图像
def random_get_subimg( img, size ):
	max_x,max_y,_ = img.shape
	point = ( np.random.randint(max_x), np.random.randint(max_y) ) # 随机生成一个点
	while(point[0]+size[0]>=max_x or point[1]+size[1]>=max_y):
		point = ( np.random.randint(max_x), np.random.randint(max_y) )
	return img[point[0]:point[0]+size[0],point[1]:point[1]+size[1],:]

# 随机生成size大小的RGB图像
def random_gen_imgs(num=100,size=(512,512)):
	tif_files = []
	for idx in file_idx:
		tif_files.append('./data/test/LC81240322017118LGN00_B%d.TIF'%(idx))
	img = []
	for i,tif_file in enumerate(tif_files,0):
		print tif_file
		img.append( tiff.imread(tif_file) )
	img = np.array(img)
	img = img.transpose([1,2,0])
	img = img*1.0/65535*256;
	img.astype(np.int)
	# plt.imshow(img[:,:,:])
	# plt.show()
	print(img.shape)
	for i in range(num):
		sub_img = random_get_subimg(img,size)
		# plt.imshow(sub_img[:,:,:])
		# plt.show()
		if not os.path.exists("./cache/gen_imgs/"): os.makedirs("./cache/gen_imgs/")
		cv2.imwrite("./cache/gen_imgs/random_gen_%.3d.png"%(i),sub_img)

if __name__ == '__main__':
	random_gen_imgs()
