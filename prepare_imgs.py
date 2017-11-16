#-*- coding:utf-8 -*-
#!/usr/bin/python
import argparse
import json
import os.path as osp
import sys
import PIL.Image
import yaml
from labelme import utils
import numpy as np
import csv, sys, cv2, os, math, shutil
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
	GPS_name = "LC81240332017118LGN00"
	tif_files = []
	for idx in file_idx:
		tif_files.append('./data/%s/%s_B%d.TIF'%(GPS_name,GPS_name,idx))
	img = []
	for i,tif_file in enumerate(tif_files,0):
		print(tif_file)
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

# labelme的关键函数从一个json文件转换成图像与其label	
def json_to_dataset(json_file_path):
    out_dir = osp.basename(json_file_path).replace('.', '_')
    out_dir = osp.join(osp.dirname(json_file_path), out_dir)
    if not osp.exists(out_dir): os.mkdir(out_dir)
    data = json.load(open(json_file_path)) 
    img = utils.img_b64_to_array(data['imageData'])
    lbl, lbl_names = utils.labelme_shapes_to_label(img.shape, data['shapes'])
    lbl_viz = utils.draw_label(lbl, img, lbl_names)
    PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
    PIL.Image.fromarray(lbl).save(osp.join(out_dir, 'label.png'))
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
    info = dict(label_names=lbl_names)
    with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
        yaml.safe_dump(info, f, default_flow_style=False)
    print('wrote data to %s' % out_dir)

# 从所有json文件中转换label.png
def Recursive_jsons():
    root_path = "./labelme_gen/jsons/"
    file_names = os.listdir(root_path)
    for i,file_name in enumerate(file_names,0):
        # if i>=1:break
        if ".json" not in file_name: continue;
        json_to_dataset(root_path+file_name)

# 制作训练数据到./cache/train_imgs/imgs和./cache/train_imgs/labels
def make_train_files():
	root_path = "./labelme_gen/jsons/"
	out_root_path = "./cache/train_imgs/"
	if not os.path.exists(out_root_path+"imgs/"): os.makedirs(out_root_path+"imgs/")
	if not os.path.exists(out_root_path+"labels/"): os.makedirs(out_root_path+"labels/")
	file_names = os.listdir(root_path)
	for i,file_name in enumerate(file_names,0):
		# if i>=1:break
		if ".json" in file_name: continue;
		img_path = root_path + file_name +"/img.png"
		label_path = root_path + file_name +"/label.png"
		img_out_path = out_root_path+"imgs/"+file_name+".png"
		label_out_path = out_root_path+"labels/"+file_name+".png"
		shutil.copy( img_path, img_out_path )
		shutil.copy( label_path, label_out_path )  

if __name__ == '__main__':
	random_gen_imgs()
	# Recursive_jsons()
	# make_train_files()
	pass













