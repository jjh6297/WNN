import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import os
import model as wnn_pt
import WNN as wnn_tf
import re

script_dir = os.path.dirname(os.path.abspath('conversion.py'))


###################################################################################################
# Functions to calculate the error between the two models' weights:

def calculate_error(matrix1, matrix2):

    error = np.linalg.norm(matrix1 - matrix2)
    return error

def calculate_error_RMSE(M1, M2):
    return np.sqrt(np.sum((M1-M2)**2))

def model_error(model_pt, model_tf, frobenius = True, RMSE = False):
    '''
    Calculate the total error between the weights of two models.\n
    This is used to check if the conversion is successful.
    '''
    err = 0
    for name, p in model_pt.named_modules():
        if get_layer(p)!=True:
            continue
        tf_ll = find_layer(model_tf, name)
        if frobenius:
            err += calculate_error(tf_ll.get_weights()[0].T, p.weight.detach().numpy())
        if RMSE:
            err += calculate_error_RMSE(tf_ll.get_weights()[0].T, p.weight.detach().numpy())
    return err

# Functions to search for the good layers:

def find_layer(model, name):
    '''
    Find the corresponding layer in the tensorflow model.\n
    This works only if the layers are named the exact same way in the two models.
    '''
    if name.endswith('.linear'):
        new_name = re.sub(r'\.linear$', '', name)
    else:
        new_name = name
    for p in model.layers:
        if p.name == new_name:
            return p
    print("Error: Layer not found")
    return -1

def get_layer(layer): # If we have a layer with weights and bias, we return True
    '''
    This function checks if we have a layer that might contain weights and bias.\n
    The purpose of this is to check if we're not in the NN instance which makes the code crash.
    '''
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        return True
    return False
###################################################################################################


def convert_weights_tf_to_pt(model_tf, model_pt, _conv = False, _bias = False, _fc= False, save=False, filepath=''): 

    '''
    This function converts the weights of a tensorflow model to a pytorch model.\n
    args:\n
        `model_tf`: tensorflow model\n
        `model_pt`: pytorch model\n
        `save`: if True, the pytorch model is saved as 'filepath'\n
        `_conv`, `_bias`, `_fc`: if True, the model is saved as 'NWNN_Conv_13.pt', 'NWNN_Bias_13.pt' or 'NWNN_FC_13.pt' respectively\n
    Returns:\n
        ``0`` if the conversion is successful\n
        ``-1`` if there is an error (occur if there's a layer not found or if the error is too big)\n
    Note: (very important)\n
    This function is made for the WNN models, so it only works for dense layers. Furthermore the dense layers have to be named
    the exact same way in the two models or else the function will either not find the layer or copy the weights in the wrong instance.
    In fact, any model following those two conditions can be converted.
    '''
    
    for name, p in model_pt.named_modules():
    
        if get_layer(p)!=True: # Mandatory to check if we're not in the NN instance which makes the code crash
            continue
        
        if name[0] == 'd': # We basically convert only FC layers based models so we only check for linear layers. 
                           # In fact checking get_layer(p) is enough to not bother using this but this code is easier to transform to convert other layers if needed.
            tf_ll = find_layer(model_tf, name) # We find the corresponding layer in the tensorflow model
            
            kernel = torch.tensor(tf_ll.get_weights()[0].T, dtype=torch.float32)
            
        
            if p.bias is not None:
                bias = torch.tensor(tf_ll.get_weights()[1], dtype=torch.float32)

            with torch.no_grad(): # We don't want to compute the gradient of the weights (or else fine tuning from freshly converted models will mess up the weights)

                p.weight.copy_(kernel)
                if p.bias is not None:
                    p.bias.copy_(bias)
                    
            err = calculate_error(tf_ll.get_weights()[0].T, kernel.numpy()) # We check the error between the two models' weights
            if err != 0: # If it's anything else 0 then there's an error
                print("Conversion failed in the layer: " + name)
                return -1
            
    if save and filepath != '':
        file_path = os.path.join(script_dir, filepath)
        torch.save(model_pt.state_dict(), file_path)
        print("Conversion done successfully")
        return 0
    if _conv:
        file_path = os.path.join(script_dir, "WNN_PT/NWNN_Conv_13.pt")
        torch.save(model_pt.state_dict(), file_path) 
        print("Conversion done successfully for conv layers")
        return 0
    if _fc:
        file_path = os.path.join(script_dir, "WNN_PT/NWNN_FC_13.pt")
        torch.save(model_pt.state_dict(), file_path) 
        print("Conversion done successfully for FC layers")
        return 0
    if _bias:
        file_path = os.path.join(script_dir, "WNN_PT/NWNN_Bias_13.pt")
        torch.save(model_pt.state_dict(), file_path) 
        print("Conversion done successfully for bias layers")
        return 0
    print("Conversion done successfully")
        
###################################################################################################
# Functions to convert the weights from pytorch to tensorflow: (opposite operation)

def convert_weight_pt_to_tf(model_pt, model_tf, save=False, filepath=''):
    '''
    This function converts the weights of a pytorch model to a tensorflow model.\n
    args:\n
        `model_pt`: pytorch model\n
        `model_tf`: tensorflow model\n
        `save`: if True, the tensorflow model is saved as 'filepath'
    Returns:\n
        ``0`` if the conversion is successful\n
        ``-1`` if there is an error (occur if there's a layer not found or if the error is too big)\n
    Note:\n
    This function is made for the WNN models, so it only works for dense layers. Furthermore the dense layers have to be named
    the exact same way in the two models or else the function will either not find the layer or copy the weights in the wrong instance.
    In fact, any model following those two conditions can be converted. Also this function was to test the conversion in the other way to check
    if the weight could change going from tensorflow to pytorch and back to tensorflow.
    
    '''
    err_tot = 0
    for name, p in model_pt.named_modules():
        if get_layer(p)!=True: 
            continue
        if name[0] == 'd':
            tf_ll = find_layer(model_tf, name)
            if tf_ll == -1:
                print("Error: Layer not found")
                return -1
            
            kernel = np.transpose(p.weight.detach().numpy())
            if p.bias is not None:
                bias = p.bias.detach().numpy()
                tf_ll.set_weights([kernel, bias])
                err_tot += calculate_error(tf_ll.get_weights()[0].T, p.weight.detach().numpy()) + calculate_error(tf_ll.get_weights()[1], p.bias.detach().numpy())
            else:
                tf_ll.set_weights([kernel])
                err_tot += calculate_error(tf_ll.get_weights()[0].T, p.weight.detach().numpy())
    if err_tot != 0: # We check if the error is anything else than 0 (then means there's a gap in the weights of the two models)
        print("Error in the conversion of the weights")
        return -1
    if save and filepath != '':
        file_path = os.path.join(script_dir, filepath)
        model_tf.save_weights(file_path)
    print("Conversion done successfully")
    return 0

###################################################################################################

def conversion_WNN(bias = False, conv = False, fc= False):
    '''
    This function converts the weights of the WNN models from tensorflow to pytorch and saves them.\n
    To make this work, you have to turn at least one of the arguments to True.\n
    '''
    tf_WNN = wnn_tf.WNN(5)
    pt_WNN = wnn_pt.WNN()
    if conv:
        file_path = os.path.join(script_dir, 'WNN_PT/NWNN_Conv_13.h5')
        tf_WNN.load_weights(file_path)
        if convert_weights_tf_to_pt(tf_WNN, pt_WNN, _conv = True) == -1:
            return -1
        file_path = os.path.join(script_dir, 'WNN_PT/NWNN_Conv_13.pt')
        pt_WNN.load_state_dict(torch.load(file_path))

    if bias:
        file_path = os.path.join(script_dir, 'WNN_PT/NWNN_Bias_13.h5')
        tf_WNN.load_weights(file_path)
        if convert_weights_tf_to_pt(tf_WNN, pt_WNN, _bias = True) == -1:
            return -1
        file_path = os.path.join(script_dir, 'WNN_PT/NWNN_Bias_13.pt')
        pt_WNN.load_state_dict(torch.load(file_path))

    if fc:
        file_path = os.path.join(script_dir, 'WNN_PT/NWNN_FC_13.h5')
        tf_WNN.load_weights(file_path)
        if convert_weights_tf_to_pt(tf_WNN, pt_WNN, _fc = True) == -1:
            return -1
        file_path = os.path.join(script_dir, 'WNN_PT/NWNN_FC_13.pt')
        pt_WNN.load_state_dict(torch.load(file_path))

    return 0