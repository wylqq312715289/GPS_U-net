#-*- coding:utf-8 -*-
import numpy as np
import pandas as pd 
import os
from easydict import EasyDict as edict

if not os.path.exists("./cache/npy/"): os.makedirs("./cache/npy/")
if not os.path.exists("./cache/model/"): os.makedirs("./cache/model/") # 存放模型的地址
if not os.path.exists("./cache/test_imgs/pred_labels/"): os.makedirs("./cache/test_imgs/pred_labels/") # 存放模型的地址

config = edict()
config.data_path = "./cache/train_imgs/imgs"
config.label_path = "./cache/train_imgs/labels"
config.test_path = "./cache/test_imgs/imgs"
config.test_pred_label_path = "./cache/test_imgs/pred_labels"
config.npy_path = "./cache/npy"

config.model_path = "./cache/model/u-net.h5"
config.img_rows = 512 
config.img_cols = 512
config.batch_size = 4  # 深度模型 分批训练的批量大小
config.epochs = 5       # 总共训练的轮数（实际不会超过该轮次，因为有early_stop限制）
config.early_stop = 3  # 最优epoch的置信epochs



