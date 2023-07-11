from __future__ import print_function
import tensorflow.keras
# from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.optimizers import  Adam
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, History
from tensorflow.keras import backend as K
from itertools import combinations
from scipy.spatial import distance
import sys
from tensorflow.keras.constraints import unit_norm
# from itertools import combinations
import h5py
import random
from tensorflow.keras.layers import Input, Dense, Permute, Reshape
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalMaxPooling2D, LeakyReLU, Reshape, Concatenate
from tensorflow.keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Lambda, Multiply, Add, add, multiply, SpatialDropout2D, GaussianDropout, AlphaDropout, GlobalAveragePooling2D
# from tensorflow.keras.utils.training_utils import multi_gpu_model
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
# import scipy.io as sio
# import math


def trAct_1D_Exp(x,compressed, numExp):

	xShape2=x.shape
	x2 = Dense(compressed, activation='relu')(x)
	temp = Dense(xShape2[1])(x2)
	for jj in range(2,numExp):
		temp = Add()([Multiply()([Dense(xShape2[1],activity_regularizer=regularizers.l2(0.0))(x2),Lambda(lambda x: K.exp(x))(Multiply()([Dense(xShape2[1],activity_regularizer=regularizers.l2(0.0))(x2),x]))]),temp])

	return temp	


    
def trAct_1D_Exp(x,compressed, numExp):

	xShape2=x.shape
	x2 = Dense(compressed, activation='relu')(x)
	temp = Dense(xShape2[1])(x2)
	for jj in range(2,numExp):
		temp = Add()([Multiply()([Dense(xShape2[1],activity_regularizer=regularizers.l2(0.0))(x2),Lambda(lambda x: K.exp(x))(Multiply()([Dense(xShape2[1],activity_regularizer=regularizers.l2(0.0))(x2),x]))]),temp])

	return temp	



reg = 1e-6
reg2 = 0.1

def WNN(NumLength=5):
    inputs = Input(shape=(NumLength,))
    inputs2 = Input(shape=(NumLength-1,))

    fc = Dense(64, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'), bias_initializer=initializers.Zeros())(inputs)
    fc = trAct_1D_Exp(fc,8,3)

    fc = Dense(32, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'), bias_initializer=initializers.Zeros())(fc)
    fc = LeakyReLU()(fc)



    fc2 = Dense(64, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'), bias_initializer=initializers.Zeros())(inputs2)
    fc2 = trAct_1D_Exp(fc2,8,3)

    fc2 = Dense(32, kernel_regularizer=regularizers.l1(reg), bias_regularizer=regularizers.l1(reg2),kernel_initializer=initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform'), bias_initializer=initializers.Zeros())(fc2)
    fc2= LeakyReLU()(fc2)



    fc = Dense(1,activation='tanh')(Concatenate()([fc,fc2]))

    network = Model([inputs,inputs2], fc)
    return network
    
    
def get_ConvLayer_pred(layer, predictor, NumLength=5):
    Layer = np.reshape(layer,[layer.shape[0]*layer.shape[1]*layer.shape[2]*layer.shape[3],layer.shape[4]])  
    dim = Layer.shape[1]
    minval = np.tile(Layer[:,0:1]  ,(1,dim)) 
    Layer=Layer -minval   
    maxval = np.tile(np.max(Layer,1, keepdims=True)-np.min(Layer,1, keepdims=True)+1e-8  ,(1, dim)) 

    Layer=Layer/(maxval+0.00001)/2.         
    return np.reshape((Layer[:,NumLength-1] +  predictor.predict([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]], batch_size = 100000)[:,0])*(maxval[:,0]+0.00001)*2. +minval[:,0], [layer.shape[0], layer.shape[1], layer.shape[2], layer.shape[3]])


def get_DepthConvLayer_pred(layer, predictor, NumLength=5):
    Layer = np.reshape(layer,[layer.shape[0]*layer.shape[1]*layer.shape[2],layer.shape[3]])       
    dim = Layer.shape[1]
    minval = np.tile(Layer[:,0:1]  ,(1,dim)) 
    Layer=Layer -minval   
    maxval = np.tile(np.max(Layer,1, keepdims=True)-np.min(Layer,1, keepdims=True)+1e-8  ,(1, dim)) 

    Layer=Layer/(maxval+0.00001)/2.       
    return (np.reshape(Layer[:,NumLength-1] +  predictor.predict([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]], batch_size = 100000)[:,0], [layer.shape[0], layer.shape[1], layer.shape[2]])  )*(maxval[:,0]+0.00001)*2. +minval[:,0]



def get_FCLayer_pred(layer, predictor, NumLength=5):
    Layer = np.reshape(layer,[layer.shape[0]*layer.shape[1], layer.shape[2]])         
    dim = Layer.shape[1]
    minval = np.tile(Layer[:,0:1]  ,(1,dim)) 
    Layer=Layer -minval   
    maxval = np.tile(np.max(Layer,1, keepdims=True)-np.min(Layer,1, keepdims=True)+1e-8  ,(1, dim)) 

    Layer=Layer/(maxval+0.00001)/2.       
    return np.reshape((Layer[:,NumLength-1] +  predictor.predict([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]], batch_size = 100000)[:,0])*(maxval[:,0]+0.00001)*2. +minval[:,0], [layer.shape[0], layer.shape[1]])

    
    
def get_BiasLayer_pred(layer, predictor, NumLength=5):
    Layer = layer
    dim = Layer.shape[1]
    minval = np.tile(Layer[:,0:1]  ,(1,dim)) 
    Layer=Layer -minval   
    maxval = np.tile(np.max(Layer,1, keepdims=True)-np.min(Layer,1, keepdims=True)+1e-8  ,(1, dim)) 

    Layer=Layer/(maxval+0.00001)/2.       
    return (layer[:,NumLength-1] +  predictor.predict([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]], batch_size = 100000)[:,0])*(maxval[:,0]+0.00001)*2. +minval[:,0]



global WeightsSet
global BiasSet
WeightsSet=[]
BiasSet=[]
class WeightForecasting(tensorflow.keras.callbacks.Callback):
    def __init__(self):
        super(WeightForecasting, self).__init__()
        self.model2 = WNN(5)
        self.model2.compile(loss='mae', optimizer=Adam(),metrics=['mae'])
        self.model2.load_weights('NWNN_Conv_13.h5')
        
        self.model3 = WNN(5)
        self.model3.compile(loss='mae', optimizer=Adam(),metrics=['mae'])
        self.model3.load_weights('NWNN_FC_13.h5')						

        self.model4 = WNN(5)
        self.model4.compile(loss='mae', optimizer=Adam(),metrics=['mae'])       
        self.model4.load_weights('NWNN_Bias_13.h5')
        self.cnt=0

        

    def on_epoch_end(self, epoch, logs=None):
        global WeightsSet, BiasSet        
        if epoch==0:
            self.cnt = 0
            for ll in self.model.layers:
                WW = ll.get_weights()
                if len(WW)>0 and len(WW)<4:
                    if len(WW)==1:
                        WeightsSet.append(np.zeros((WW[0].shape+(5,))))
                        BiasSet.append([])
                        
                    else: 	
                        WeightsSet.append(np.zeros((WW[0].shape+(5,))))
                        BiasSet.append(np.zeros((WW[1].shape+(5,))))
                    self.cnt=self.cnt+1	
            print('Init')
        idx = epoch%5
                  
        
        self.cnt = 0
        for ll in self.model.layers:
            WW = ll.get_weights()

            if len(WW)>0 and len(WW)<4:
                if len(WW)==1:
                    if len(WW[0].shape)==4:
                        WeightsSet[self.cnt][:,:,:,:,idx] = WW[0]
                    elif len(WW[0].shape)==3:
                        WeightsSet[self.cnt][:,:,:,idx] = WW[0]         
                    else:
                        WeightsSet[self.cnt][:,:,idx] = WW[0]                                

                
                else:
                    if len(WW[0].shape)==4:
                        WeightsSet[self.cnt][:,:,:,:,idx] = WW[0]
                    elif len(WW[0].shape)==3:
                        WeightsSet[self.cnt][:,:,:,idx] = WW[0]         
                    else:
                        WeightsSet[self.cnt][:,:,idx] = WW[0]        
                    BiasSet[self.cnt][:,idx] = WW[1]

                self.cnt=self.cnt+1	
        if epoch%5==4 and epoch>0:
            self.cnt = 0
            for ll in self.model.layers:
                WW = ll.get_weights()
                if len(WW)>0 and len(WW)<4:
                    if len(WW)==1:
                        if len(WW[0].shape)==4:
                            NewWeights = np.array(get_ConvLayer_pred(WeightsSet[self.cnt],self.model2,5))
                        if len(WW[0].shape)==3:
                            NewWeights = np.array(get_DepthwiseConvLayer_pred(WeightsSet[self.cnt],self.model2,5))
                        if len(WW[0].shape)==2:
                            NewWeights = np.array(get_FCLayer_pred(WeightsSet[self.cnt],self.model3,5))
                        ll.set_weights([NewWeights])						                    
                    else: 	
                        if len(WW[0].shape)==4:
                            NewWeights = np.array(get_ConvLayer_pred(WeightsSet[self.cnt],self.model2,5))
                        if len(WW[0].shape)==3:
                            NewWeights = np.array(get_DepthwiseConvLayer_pred(WeightsSet[self.cnt],self.model2,5))
                        if len(WW[0].shape)==2:
                            NewWeights = np.array(get_FCLayer_pred(WeightsSet[self.cnt],self.model3,5))
                        NewBias = np.array(get_BiasLayer_pred(BiasSet[self.cnt],self.model4,5))
                        ll.set_weights([NewWeights, NewBias])		                  
                    self.cnt=self.cnt+1
					
            print('Forecasting Done')