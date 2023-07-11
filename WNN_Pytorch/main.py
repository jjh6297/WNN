import conversion
import model as WNN_pytorch
import WNN
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import input_test

def main():
    # input_test.input_test(X1_tf = 0, X1_pt = 0, fc = False, conv = False, bias = False, random = False)
    return conversion.conversion_WNN(bias= True, conv= True, fc= True)

if __name__ == "__main__":
    main()