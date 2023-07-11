import torch
import torch.nn as nn


input_size = 5
output_size = 1
batch_size = 64
learning_rate = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Definition of the layers we need in the network :

class LinearL1(nn.Module):
    
    def __init__(self, in_features, out_features):
        super(LinearL1, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.reg_weight_decay = 1e-6
        self.reg2_weight_decay = 0.1
        self.register_buffer('weight_loss', torch.zeros(1))
        self.register_buffer('bias_loss', torch.zeros(1))

    def forward(self, x):
        output = self.linear(x)

        # Ajouter la régularisation L1 aux poids
        self.weight_loss = torch.norm(self.linear.weight, p=1)
        self.bias_loss = torch.norm(self.linear.bias, p=1)
        regularizer_loss = self.reg_weight_decay * self.weight_loss + self.bias_loss*self.reg2_weight_decay
        
        return output, regularizer_loss


# Lambda functions wrapped in a layer to be able to use them in the network
# We need them to apply exp function to an output of a layer in the network
class LambdaLayer(nn.Module):
    def __init__(self, lambda_func):
        super(LambdaLayer, self).__init__()
        self.lambda_func = lambda_func
        
    def forward(self, x):
        return self.lambda_func(x)
    

# Multiplication layer to multiply two layers :

class MultiplyLayer(nn.Module):
    def __init__(self):
        super(MultiplyLayer, self).__init__()
    
    def forward(self, inputs):
        assert len(inputs) == 2, "MultiplyLayer expects exactly 2 inputs"
        return torch.mul(inputs[0], inputs[1])

# Addition layer to add two layers :

class AddLayer(nn.Module):
    def __init__(self):
        super(AddLayer, self).__init__()
    
    def forward(self, inputs):
        assert len(inputs) == 2, "AddLayer expects exactly 2 inputs"
        return torch.add(inputs[0], inputs[1])
    
# Concatenation layer to concatenate two layers :

class ConcatLayer(nn.Module):
    def __init__(self):
        super(ConcatLayer, self).__init__()
        
    def forward(self, inputs):
        assert len(inputs) == 2, "ConcatLayer expects exactly 2 inputs"
        return torch.concat((inputs[0], inputs[1]), -1)
    
# Note that we don't need to define a constructor due to the fact there's no need to initialize any layers like in the lambda layer

class WNN(nn.Module):
    '''
    This is the WNN model written in PyTorch. It is a weight prediction model based on the Tensorflow version of the WNN.
    It follows the exact same architecture and layer name (that you can check on the WNN.summary() of Tensorflow).
    
    The model work with the function implemented in the ``WNN_PT/wnn_callback.py`` file which is the exact same function used in the 
    WeightForecasting callback class of the Tensorflow version.
    
    You can also check the architecture in the ``WNN_PT/Viz/model_pt.onnx.png`` and the ``WNN_PT/Viz/model_tf.onnx.png``which represents the two versions of the WNN
    written in Tensorflow and PyTorch (it is the exact same architecture but slightly different because of the framework)
    '''
    def __init__(self):
        
        super(WNN, self).__init__()
         
        # We're defining the differents layers of the network here :
        # In fact, there's only 12 differents layer throughout the network (6 dense, 4 calculus layers, the LeakyReLU layer and the ReLU one)
        
        
        # Init with torch.nn.init to implement the same initialization as in the Tensorflow version
        
        self.dense = LinearL1(input_size, 64) # input 1 (dense n°0)
        self.dense_6 = LinearL1(input_size-1, 64) # input 2 (dense n°6)
        
        self.dense_1 = nn.Linear(64, 8) 
        self.dense_2 = nn.Linear(8, 64) 
        self.dense_3 = nn.Linear(8, 64)
        self.dense_4 = nn.Linear(8, 64) 
        self.dense_5 = LinearL1(64,32) 

        self.dense_7 = nn.Linear(64, 8) 
        self.dense_8 = nn.Linear(8, 64) 
        self.dense_9 = nn.Linear(8, 64) 
        self.dense_10 = nn.Linear(8, 64) 
        self.dense_11 = LinearL1(64, 32)
        
        self.dense_12 = nn.Linear(64, output_size) 
        
        self.lambda_0 = LambdaLayer(lambda x: torch.exp(x)) 
        self.lambda_1 = LambdaLayer(lambda x: torch.exp(x)) 
        self.add = AddLayer() 
        self.add_1 = AddLayer() 
        self.multiply = MultiplyLayer() 
        self.multiply_1 = MultiplyLayer() 
        self.multiply_2 = MultiplyLayer() 
        self.multiply_3 = MultiplyLayer()
        
        self.leaky_re_lu = nn.LeakyReLU() 
        self.leaky_re_lu_1 = nn.LeakyReLU()
        
        self.concatenate = ConcatLayer() 
        
        
        # We have every layer we need for the forward propagation
        # Note : We directly hard coded the size of inputs and outputs of each layer --> We don't need to pay attention to
        #                                                                                the batch size pytorch will do it for us
        
        
    def forward(self, x):
        
        x1 = x[0]
        x2 = x[1]
        # x2 = x1[0:4] - x1[1:5]
        
        fc1 = self.dense(x1)[0] # dense n°0
        d1 = self.dense_1(fc1) 
        d1 = nn.ReLU()(d1) 
        d2 = self.dense_2(d1)
        d3 = self.dense_3(d1)
        d4 = self.dense_4(d1)
        
        lambd0 = self.lambda_0(self.multiply_1([fc1, d4])) 
        d5 = self.dense_5(self.add([self.multiply([d3, lambd0]), d2]))[0]
        leaky = self.leaky_re_lu(d5)
        
        fc2 = self.dense_6(x2)[0] # dense n°6
        d7 = self.dense_7(fc2)
        d7 = nn.ReLU()(d7)
        d8 = self.dense_8(d7)
        d9 = self.dense_9(d7)
        d10 = self.dense_10(d7)
        
        lambda1 = self.lambda_1(self.multiply_3([fc2, d10]))
        d11 = self.dense_11(self.add_1([self.multiply_2([d9, lambda1]), d8]))[0]
        leaky1 = self.leaky_re_lu_1(d11)
        
        conc = self.concatenate([leaky, leaky1])
        out = self.dense_12(conc)
        out = nn.Tanh()(out)
        
        return out 
    

# Training loop (wrapped in a function) :
    
def training(model, trainloader, save=False, filepath=None):
    '''
    This is the training function which contain the training loop of the model.
    
    You can save the model by setting the save argument to True and by specifying the filepath where you want to save the model. (doesn't work if you only set save to True)
    '''
    epochs = 100
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for i, (input, gt) in enumerate(trainloader):
            
            optimizer.zero_grad()
            outputs, regularization_loss = model(input)
            loss_value = loss(outputs, gt)
            loss_value += regularization_loss
            loss_value.backward()
            optimizer.step()
            
            if i % 1000 == 0:
                print(f"Epoch : {epoch} | Loss : {loss_value.item()}")
    
    if save and filepath is not None:
        torch.save(model.state_dict(), filepath)