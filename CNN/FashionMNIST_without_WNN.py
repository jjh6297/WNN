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
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.regularizers import l1, l2
# from WNN import *

import copy
sys.setrecursionlimit(10000)
nb_classes = 10
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = np.reshape(x_train,[x_train.shape[0], 28, 28, 1])
x_test = np.reshape(x_test,[x_test.shape[0], 28, 28, 1])
input_shape = x_train.shape[1:]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean


y_train = tensorflow.keras.utils.to_categorical(y_train, nb_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, nb_classes)
lll = 0.0


	

trainableLabel = True

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):

    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  
                strides = 2  
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tensorflow.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    y = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)
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
	model = resnet_v1(input_shape=(28,28,1), depth=32)
	adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	batch_size = 1024
	model.compile(loss='categorical_crossentropy',
				  optimizer=adam,
				  metrics=['accuracy'])	
						   

	Loss = []
	Acc = []
	ValLoss = []
	ValAcc = []
	hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=int(x_train.shape[0]/ batch_size),epochs=50, verbose=1, workers=1,validation_data = (x_test, y_test))
	Loss = np.array(hist.history['loss'])
	ValLoss = np.array(hist.history['val_loss'])
	Acc = np.array(hist.history['accuracy'])
	ValAcc = np.array(hist.history['val_accuracy'])
	sio.savemat('Without_WNN_trial_'+str(trial)+'_FashionMNIST.mat', mdict = {'Loss':Loss, 'ValLoss':ValLoss, 'Acc':Acc, 'ValAcc':ValAcc})
	K.clear_session()
