# WNN PyTorch version :

This is the WNN written in PyTorch, you can find the different file needed to convert the model from the Tensorflow version to the PyTorch one and the file needed to use the architecture.

## Usage example:

```python
import torch
import torch.nn as nn
import wnn_callback as wnn

# Data processing and hyper-parameters:

learning_rate = 0.001
batch_size = 100
...

train_dataset = ...
test_dataset = ...

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)



# Define your NN:

class NeuralNetwork(nn.Module):

    def __init__(self, ...):
        super(NeuralNetwork, self).__init__()
        ...

    def forward(self, x):
        ...

        return out

# Training loop:

def train(model, epochs):

    loss = ...
    optimizer = ...

    for epoch in range(epochs):
        for i, (x,y) in enumerate(train_loader):

            output = model(x)
            l = loss(output,y)

            l.backward()
            optimizer.step()
            optimizer.zero_grad()

        # At this point of the training loop, we finished the backward propagation and weights update so we can use the WNN here:
        wnn.WeightForecasting(model, epoch)

    return None


```

In the file containing the ```WeightForecasting()``` function, you can find another function that uses the Tensorflow WNN to predict PyTorch weights. The usage is exactly the same as shown above and uses the original WNN model.

----

<u>Notes</u>:

Within the repo, you can also find the functions to convert the weights in the [conversion.py](/WNN_PT/conversion.py) file. You can use them by running a script such as (it's also in the main function in [main.py](/WNN_PT/main.py) ):

```python
import conversion

conversion.conversion_WNN(bias = True)
```

You can either turn the option `bias`, `conv`, or `fc` to `True`. This function is however written so you can **only convert weights if the two models have the same layers' name and use only fully-connected layers** as it is intended to convert the WNN which only contain such layer.

You can also use the [onnx_export.py](/WNN_PT/onnx_export.py) file to export the model to ONNX. The code structure is very simple and can be used to run the model on onnx runtime or to vizualise it thanks Netron. By going on https://netron.app you can open your onnx model to have your model displayed as a very readable and easy to understand model graph. You can check the results in the <span style="color:#0366d6">Viz</span> folder. (You can note that the PyTorch model follows exactly the same pattern as the Tensorflow WNN as expected)



Finally, you there is a input/output testing function in the [input_test.py](/WNN_PT/input_test.py) file. This function can be used easily in [main.py](/WNN_PT/main.py) by uncommenting the code in the main function. You can use the different arguments the way you want to see the differences between the two models in the two frameworks. This function is not intended to work with NN but with only weights from one layer.