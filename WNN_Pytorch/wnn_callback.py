import model as md
import torch
import torch.nn as nn


def get_ConvLayer_pred(layer, predictor, NumLength=5):
    Layer = torch.reshape(layer, [layer.shape[0]*layer.shape[1]*layer.shape[2]*layer.shape[3],layer.shape[4]])
    dim = Layer.shape[1]
    minval = torch.tile(Layer[:,0:1], (1,dim))
    Layer=Layer -minval   
    maxval = torch.tile(torch.max(Layer,1, keepdims=True)[0]-torch.min(Layer,1, keepdims=True)[0]+1e-8  ,(1, dim)) 

    Layer=Layer/(maxval+0.00001)/2.         
    return torch.reshape((Layer[:,NumLength-1] +  predictor([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]])[:,0])*(maxval[:,0]+0.00001)*2. +minval[:,0], [layer.shape[0], layer.shape[1], layer.shape[2], layer.shape[3]])


def get_DepthConvLayer_pred(layer, predictor, NumLength=5):
    Layer = torch.reshape(layer,[layer.shape[0]*layer.shape[1]*layer.shape[2],layer.shape[3]])       
    dim = Layer.shape[1]
    minval = torch.tile(Layer[:,0:1]  ,(1,dim)) 
    Layer=Layer -minval   
    maxval = torch.tile(torch.max(Layer,1, keepdims=True)[0]-torch.min(Layer,1, keepdims=True)[0]+1e-8  ,(1, dim)) 

    Layer=Layer/(maxval+0.00001)/2.
    return (torch.reshape(Layer[:,NumLength-1] +  predictor([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]])[:,0], [layer.shape[0], layer.shape[1], layer.shape[2]])  )*(maxval[:,0]+0.00001)*2. +minval[:,0]



def get_FCLayer_pred(layer, predictor, NumLength=5):
    Layer = torch.reshape(layer,[layer.shape[0]*layer.shape[1], layer.shape[2]])
    dim = Layer.shape[1]
    minval = torch.tile(Layer[:,0:1]  ,(1,dim)) 
    Layer=Layer -minval   
    maxval = torch.tile(torch.max(Layer,1, keepdims=True)[0]-torch.min(Layer,1, keepdims=True)[0]+1e-8  ,(1, dim)) 

    Layer=Layer/(maxval+0.00001)/2.       
    return torch.reshape((Layer[:,NumLength-1] +  predictor([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]])[:,0])*(maxval[:,0]+0.00001)*2. +minval[:,0], [layer.shape[0], layer.shape[1]])


def get_BiasLayer_pred(layer, predictor, NumLength=5):
    Layer = layer
    dim = Layer.shape[1]
    minval = torch.tile(Layer[:,0:1]  ,(1,dim)) 
    Layer=Layer -minval   
    maxval = torch.tile(torch.max(Layer,1, keepdims=True)[0]-torch.min(Layer,1, keepdims=True)[0]+1e-8  ,(1, dim)) 

    Layer=Layer/(maxval+0.00001)/2.       
    return (layer[:,NumLength-1] +  predictor([Layer, Layer[:,1:NumLength] - Layer[:,0:NumLength-1]])[:,0])*(maxval[:,0]+0.00001)*2. +minval[:,0]

global WeightsSet
global BiasSet
WeightsSet=[]
BiasSet=[]

def WNN_init():

    model2 = md.WNN()
    model2.load_state_dict(torch.load('WNN_PT/NWNN_Conv_13.pt'))
    model3 = md.WNN()
    model3.load_state_dict(torch.load('WNN_PT/NWNN_FC_13.pt'))
    model4 = md.WNN()
    model4.load_state_dict(torch.load('WNN_PT/NWNN_Bias_13.pt'))

    return model2, model3, model4

def get_layer(layer): # If we have a layer with weights

    '''
    Verify if the layer has weights in it by checking if it is an instance of nn.Conv2d, nn.ConvTranspose2d, nn.Linear or nn.BatchNorm2d.
    
    '''

    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d) or isinstance(layer, nn.Linear) or isinstance(layer, nn.BatchNorm2d):
        return True
    return False

def WeightForecasting(model, epoch):
    '''

    This is the prediction function. It first verify the epoch we're in to decide if we only store weights or make a prediction.
    The function has to verify the type of layer used to store and predict the weights properly.\n
    For instance, the shape of a Linear layer's weight is ``(out_features, in_features)`` and the shape of a Conv2d layer's weight is ``(out_channels, in_channels, kernel_size, kernel_size)``.

    Once it stored 5 epochs of weights, it can make a prediction using the stored weights and the predictor model (the WNN with the proper weights). 

    '''
    model2, model3, model4 = WNN_init()
    global WeightsSet, BiasSet
    if epoch==0: 

        for ll in model.modules():
            if get_layer(ll): # We verify if we have a layer with weights and bias
                WeightsSet.append(torch.zeros((ll.weight.shape+(5,))))
                if ll.bias == None:
                    BiasSet.append(torch.tensor([])) # On a pas eu le bias de la couche précédente donc il faut rajouter un élément vide
                else:
                    BiasSet.append(torch.zeros((ll.bias.shape+(5,))))

        print('Init')
    idx = epoch%5
    cnt = 0 # Indice de la couche dans BiasSet et WeightsSet
    for ll in model.modules():

        if get_layer(ll):

            if len(ll.weight.shape)==4:
                WeightsSet[cnt][:,:,:,:,idx] = ll.weight #  On a rajouté précedemment une dimension avec .shape + (5,) pour pouvoir stocker les 5 poids
            # elif len(ll.weight.shape)==3:
            #     WeightsSet[cnt][:,:,:,idx] = ll
            else:
                WeightsSet[cnt][:,:,idx] = ll.weight
            if ll.bias != None:
                BiasSet[cnt][:,idx] = ll.bias
            cnt = cnt+1 

    if epoch%5==4:
        cnt = 0
        for name, ll in model.named_modules(): # We use the generator 'named_modules' to get the name of the layer we need to modify the weights
            if get_layer(ll):
                with torch.no_grad():
                    if len(ll.weight.shape)==4:
                        NewWeights = get_ConvLayer_pred(WeightsSet[cnt],model2,5)
                    # if len(ll.weight.shape)==3:
                    #     NewWeights = get_DepthwiseConvLayer_pred(WeightsSet[cnt],model2,5) # On a pas de quoi implémenter ni tester cette partie
                    if len(ll.weight.shape)==2:
                        NewWeights = get_FCLayer_pred(WeightsSet[cnt],model3,5)
                    
                    exec('model.' + name + '.weight.copy_(NewWeights)')
                    
                    if ll.bias != None:
                        NewBias = get_BiasLayer_pred(BiasSet[cnt],model4,5)
                        exec('model.' + name + '.bias.copy_(NewBias)')
                cnt = cnt+1 # If we were supposed to have a bias before (but didn't get it), we increment the counter
        print('Forecasting Done')
        
        
import WNN
import tensorflow as tf
import numpy as np



def WNN_prediction(model, epoch):
    
    '''
    This is a function based on the WeightForecasting function. This function is a mix between WNN_prediction() and the class WeightForecasting of the Tensorflow WNN
    which mean that it is intended to work with the WNN written in Tensorflow to predict weight from PyTorch's model. (This is quite new so I couldn't test the 
    performance between this function and the callback function for Tensorflow models but the prediction looks good)
    
    Its use is the same as the WeightForecasting() function for PyTorch model meaning you just have to call the function at the end of your training loop. 
    
    '''
    
    model2 = WNN.WNN(5)
    model2.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(),metrics=['mae'])
    model2.load_weights(filepath='WNN_PT/NWNN_Conv_13.h5')
    
    model3 = WNN.WNN(5)
    model3.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(),metrics=['mae'])
    model3.load_weights('WNN_PT/NWNN_FC_13.h5')						

    model4 = WNN.WNN(5)
    model4.compile(loss='mae', optimizer=tf.keras.optimizers.Adam(),metrics=['mae'])       
    model4.load_weights('WNN_PT/NWNN_Bias_13.h5')
    cnt=0
    
    global WeightsSet, BiasSet   
            
    if epoch==0: 

        for ll in model.modules():
            if isinstance(ll, nn.Conv2d): # We verify if we have a layer with weights and bias
                shp = (ll.weight.shape[2], ll.weight.shape[3], ll.weight.shape[1], ll.weight.shape[0])
                WeightsSet.append(np.zeros((shp +(5,))))
            if isinstance(ll, nn.Conv2d) or isinstance(ll, nn.Linear):
                if ll.bias == None:
                    BiasSet.append(np.array([])) # On a pas eu le bias de la couche précédente donc il faut rajouter un élément vide
                else:
                    BiasSet.append(np.zeros((ll.bias.shape+(5,))))
            if isinstance(ll, nn.Linear):
                shp = (ll.weight.shape[1], ll.weight.shape[0])
                WeightsSet.append(np.zeros((shp+(5,))))

        print('Init')
    idx = epoch%5
    cnt = 0 # Indice de la couche dans BiasSet et WeightsSet
    for ll in model.modules():

        if isinstance(ll, nn.Conv2d) or isinstance(ll, nn.Linear): # On compte pas les LinearL1 ou L2 (qui sont techniquement des NN et pas des layers)

            if len(ll.weight.shape)==4: # Toujours valide en PyTorch mais la shape fait référence à (out_channels, in_channels, kernel_height, kernel_width)
                                        # En TF ce sera plutôt (kernel_height, kernel_width, in_channels, out_channels) ainsi il faudra bloquer les deux dernières dimensions                        
                WeightsSet[cnt][:,:,:,:,idx] = np.transpose(ll.weight.detach().numpy(), (2, 3, 1, 0)) # On transpose pour avoir la même shape qu'en TF
                                                                                                      # La perm dans l'autre sens est (3, 2, 0, 1)
            # elif len(ll.weight.shape)==3:
            #     WeightsSet[cnt][:,:,:,idx] = ll
            else:
                print(WeightsSet[cnt][:,:,idx].shape)
                print(ll.weight.detach().numpy().shape)
                WeightsSet[cnt][:,:,idx] = tf.transpose(ll.weight.detach().numpy())
            if ll.bias != None:
                BiasSet[cnt][:,idx] = ll.bias.detach().numpy()
            cnt = cnt+1 

    if epoch%5==4:
        cnt = 0
        for name, ll in model.named_modules(): # We use the generator 'named_modules' to get the name of the layer we need to modify the weights
            if isinstance(ll, nn.Conv2d) or isinstance(ll, nn.Linear):
                with torch.no_grad():
                    if len(ll.weight.shape)==4:
                        NewWeights = WNN.get_ConvLayer_pred(WeightsSet[cnt],model2,5)
                        NewWeights = torch.tensor(np.transpose(NewWeights, (3, 2, 0, 1))) # On transpose pour avoir la même shape qu'en PyTorch
                    # if len(ll.weight.shape)==3:
                    #     NewWeights = get_DepthwiseConvLayer_pred(WeightsSet[cnt],model2,5) # On a pas de quoi implémenter ni tester cette partie
                    if len(ll.weight.shape)==2:
                        NewWeights = WNN.get_FCLayer_pred(WeightsSet[cnt],model3,5)
                        NewWeights = torch.tensor(np.transpose(NewWeights))
                    
                    exec('model.' + name + '.weight.copy_(NewWeights)')
                    
                    if ll.bias != None:
                        NewBias = WNN.get_BiasLayer_pred(BiasSet[cnt],model4,5)
                        NewBias = torch.tensor(NewBias)
                        exec('model.' + name + '.bias.copy_(NewBias)')
                cnt = cnt+1 # If we were supposed to have a bias before (but didn't get it), we increment the counter
        print('Forecasting Done')