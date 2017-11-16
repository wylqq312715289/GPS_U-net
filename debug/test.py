#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import os,shutil
import cv2

def rename_file():
	root_path = "../labelme_gen/imgs/"
	file_names = os.listdir(root_path)
	for i,file_name in enumerate(file_names,0):
		to_file = "LC81240322017118LGN00_split_%.3d.png"%(i)
		shutil.copy(root_path + file_name,root_path + to_file)


if __name__ == '__main__':
	# # rename_file()
	# pass 
	a = np.array([[0.4,0.5],[0.6,0.3],[0.9,0.6]])
	a[a>=0.5] = 1.0
	a[a<0.5] = 0.0
	print(a)



	