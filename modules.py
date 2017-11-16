#-*- coding:utf-8 -*-
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os, cv2, glob
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.layers import concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras import backend as keras

from config import config

# 读取图像数据
class dataProcess(object):
	def __init__(self, out_rows, out_cols, img_type):
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = config.data_path
		self.label_path = config.label_path
		self.test_path = config.test_path
		self.npy_path = config.npy_path
		self.img_type = img_type

	# 读取训练图像数据存放到npy中
	def create_train_data(self):
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.data_path+"/*."+self.img_type)
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for i,imgname in enumerate(imgs, 0):
			midname = imgname[imgname.rindex("/")+1:] # 获取训练文件名称
			img = cv2.imread(self.data_path + "/" + midname,flags=cv2.IMREAD_UNCHANGED)
			label = cv2.imread(self.label_path + "/" + midname,flags=cv2.IMREAD_UNCHANGED)
			if label.shape[0]!=512:
				img = cv2.resize(img, (512, 512),interpolation=cv2.INTER_CUBIC)
				label = cv2.resize(label,(512,512),interpolation=cv2.INTER_CUBIC)
			print( img.shape, label.shape )
			imgdatas[i] = img.reshape(512,512,3)
			imglabels[i] = label.reshape(512,512,1)
			if i % 100 == 0: print('Done: {0}/{1} images'.format(i, len(imgs)))
		print('loading done')
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_label.npy', imglabels)
		print('Saving to .npy files done.')

	# 读取测试图像数据存放到npy中
	def create_test_data(self):
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs = glob.glob(self.test_path+"/*."+self.img_type)
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,3), dtype=np.uint8)
		for i,imgname in enumerate(imgs,0):
			midname = imgname[imgname.rindex("/")+1:]
			img = cv2.imread(self.test_path + "/" + midname, flags=cv2.IMREAD_UNCHANGED)
			imgdatas[i] = img.reshape(512,512,3)
		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_label = np.load(self.npy_path+"/imgs_label.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_label = imgs_label.astype('float32')
		imgs_train /= 255
		imgs_label[imgs_label > 0.5] = 1
		imgs_label[imgs_label <= 0.5] = 0
		return imgs_train, imgs_label

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		# 获取test集合中图像的文件名
		imgs = glob.glob(self.test_path+"/*."+self.img_type)
		fun = lambda x: x[x.rindex("/")+1:]
		files_name = list(map(fun,imgs))
		return imgs_test, files_name

class myUnet(object):
	def __init__(self, img_rows = 512, img_cols = 512):
		self.img_rows = img_rows
		self.img_cols = img_cols

	def get_unet(self):
		inputs = Input((self.img_rows, self.img_cols,3))

		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		print( "conv1 shape:",conv1.shape )
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		print( "conv1 shape:",conv1.shape )
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print( "pool1 shape:",pool1.shape )

		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		print( "conv2 shape:",conv2.shape )
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		print( "conv2 shape:",conv2.shape )
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print( "pool2 shape:",pool2.shape )

		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		print( "conv3 shape:",conv3.shape )
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		print( "conv3 shape:",conv3.shape )
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print( "pool3 shape:",pool3.shape )

		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		# merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
		merge6 = concatenate([drop4,up6], axis=3)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		# merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
		merge7 = concatenate([conv3,up7], axis=3)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		# merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
		merge8 = concatenate([conv2,up8], axis=3)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		# merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
		merge9 = concatenate([conv1,up9], axis=3)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
		model = Model(inputs = inputs, outputs = conv10)
		model.compile(
			optimizer = Adam(lr = 1e-5), 
			loss = 'binary_crossentropy', 
			metrics = ['accuracy'], 
		)
		return model

	def train(self, trian_x, trian_y, vali_x, vali_y ):
		# if os.path.exists(config.model_path): os.remove(config.model_path)
		if os.path.exists(config.model_path): 
			self.model = load_model(config.model_path)
		else:
			self.model = self.get_unet()
			model_checkpoint = ModelCheckpoint(filepath=config.model_path, monitor='val_loss',verbose=1, save_best_only=True)
			model_earlystop = EarlyStopping(monitor='val_loss', patience=config.early_stop, verbose=1 )
			print('Fitting model...')
			self.model.fit(
				trian_x, 
				trian_y, 
				batch_size = config.batch_size, 
				nb_epoch = config.epochs, 
				verbose = 1,
				validation_data = (vali_x, vali_y),
				shuffle=True, 
				callbacks=[ model_checkpoint, model_earlystop ],
			)

	def predict(self,test_x):
		test_y = self.model.predict(test_x, batch_size=1, verbose=1)
		test_y = np.array(test_y)
		print(np.min(test_y),np.max(test_y))
		# test_y[test_y>=0.5] = 1.0
		# test_y[test_y<0.5] = 0.0
		print(test_y.shape)
		return test_y

if __name__ == "__main__":
	#aug = myAugmentation()
	#aug.Augmentation()
	#aug.splitMerge()
	#aug.splitTransform()
	mydata = dataProcess(512,512)
	mydata.create_train_data()
	mydata.create_test_data()
	#imgs_train,imgs_label = mydata.load_train_data()
	#print( imgs_train.shape,imgs_label.shape
