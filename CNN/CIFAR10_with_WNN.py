from __future__ import print_function
import tensorflow.keras
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, History
from tensorflow.keras import backend as K
from itertools import combinations
from scipy.spatial import distance
import sys
import h5py
import random
from tensorflow.keras.layers import Input, Dense, Permute, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalMaxPooling2D, LeakyReLU, Reshape, Concatenate
from tensorflow.keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Lambda, Multiply, Add, add, multiply, SpatialDropout2D, GaussianDropout, AlphaDropout, GlobalAveragePooling2D
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy.io as sio
from WNN import *

import copy
sys.setrecursionlimit(10000)
nb_classes = 10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
input_shape = x_train.shape[1:]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean


y_train = tensorflow.keras.utils.to_categorical(y_train, nb_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, nb_classes)
lll = 0.0


	
def SimpleNet():
	inputs = Input((32, 32, 3,))

	x1 = Conv2D(8, (3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.), name='conv1')(inputs)
	x = MaxPooling2D((2, 2))(x1)

	x2 = Conv2D(16, (3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.), name='conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x2)

	x3 = Conv2D(32, (3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.), name='conv5')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x3)

	x4 = Conv2D(32, (3, 3), padding='same', activation='relu',kernel_regularizer=regularizers.l2(0.), name='conv8')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2))(x4)

	x = Flatten()(x)
	dense = Dense(64, activation='relu', name='dense1')(x)
	x = Dense(10, name='dense5')(dense)
	x = Activation("softmax")(x)


	model = Model(inputs, [x])
	return model

	

datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.,
        zoom_range=0.3,
        fill_mode='nearest',
        cval=0.,
)

					
for trial in range(5):
	model = SimpleNet()
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	batch_size = 1024
	model.compile(loss='categorical_crossentropy',
				  optimizer=adam,
				  metrics=['accuracy'])	
						   

	Loss = []
	Acc = []
	ValLoss = []
	ValAcc = []
	hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=int(x_train.shape[0]/ batch_size),epochs=50, verbose=1, workers=1,validation_data = (x_test, y_test) , callbacks=[WeightForecasting()])
	Loss = np.array(hist.history['loss'])
	ValLoss = np.array(hist.history['val_loss'])
	Acc = np.array(hist.history['accuracy'])
	ValAcc = np.array(hist.history['val_accuracy'])
	sio.savemat('With_WNN_trial_'+str(trial)+'.mat', mdict = {'Loss':Loss, 'ValLoss':ValLoss, 'Acc':Acc, 'ValAcc':ValAcc})
	K.clear_session()
