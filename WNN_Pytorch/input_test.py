import torch
import torch.nn as nn
import numpy as np
import model as WNN_pytorch
import WNN 


def input_test(X1_tf = 0, X1_pt = 0, fc = False, conv = False, bias = False, random = False):
    '''
    args:
    
        ``X1_tf``: input of the tensorflow model
        
        ``X1_pt``: input of the pytorch model
        
        ``fc``, ``conv``, ``bias``: if True, the test will be done on the corresponding model (can't use more than one at the same time)
        
        ``random``: if True, the test will be done on random variables
    
    Notes:
    
    This function shows the input / output of the models in Tensorflow and Pytorch so we can compare them.
    
    You can chose the input you want to test by changing the `X1_tf` and `X_pt` variables. If nothing is given, the input will be `[1,2,3,4,5]`.
    
    You can also chose the model you want to test by enabling the corresponding weights to `True`.
    
    Finally, you can also chose to test random variable by turning the random variable to `True`.
    '''


    tf_WNN = WNN.WNN()
    pt_WNN = WNN_pytorch.WNN()
    if bias:
        tf_WNN.load_weights('WNN_PT/NWNN_Bias_13.h5')
        pt_WNN.load_state_dict(torch.load('WNN_PT/NWNN_Bias_13.pt'))
    if conv:
        tf_WNN.load_weights('WNN_PT/NWNN_Conv_13.h5')
        pt_WNN.load_state_dict(torch.load('WNN_PT/NWNN_Conv_13.pt'))
    if fc:
        tf_WNN.load_weights('WNN_PT/NWNN_FC_13.h5')
        pt_WNN.load_state_dict(torch.load('WNN_PT/NWNN_FC_13.pt'))
        
    if X1_pt == 0 and X1_tf == 0:
        X1_tf = np.array([[1.], [2.], [3.], [4.], [5.]])
        X1_tf = np.reshape(X1_tf, (1,5))
        X1_pt = torch.tensor([1.,2.,3.,4.,5.])
        X1_pt = torch.reshape(X1_pt, (5,))
    
    if random:
        X1_tf = np.random.rand(1,5)
        X1_pt = torch.rand(5,)
    
    X2_tf = X1_tf[:,0:4] - X1_tf[:,1:5]
    X2_tf = np.reshape(X2_tf, (1,4))
    print("TF shape:")
    print(X1_tf.shape)
    print(X2_tf.shape)
    # print(X2_tf)
    
    X2_pt = X1_pt[0:4] - X1_pt[1:5]
    X2_pt = torch.reshape(X2_pt, (4,))
    print("PT shape:")
    print(X1_pt.shape)
    print(X2_pt.shape)
    
    Y_tf = tf_WNN([X1_tf, X2_tf])
    Y_pt = pt_WNN([X1_pt, X2_pt])

    print("\nTF output:")
    print(Y_tf)
    print("\nPT output:")
    print(Y_pt)