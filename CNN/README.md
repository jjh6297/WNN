
# Weight Nowcasting Network (WNN)

Code for ["Learning to Boost Training by Periodic Nowcasting Near Future Weights"]


### dependency


<!-- dependencies: -->

| Library | Known Working | Known Not Working |
| tensorflow | 2.3.0, 2.9.0 | <= 2.0 |
<!-- | tensorflow | 2.3.0, 2.4.1 | <= 2.0 | -->


### Usage
WNN can be easily used as a callback function extending tf.keras.callbacks.Callback:
```python
import tensorflow as tf
import tensorflow.keras
from WNN import *

.
.
.

model.fit(..., callbacks=[WeightForecasting()])
```
## Pre-trained Weights
Pre-trained weights of WNN are included.
'NWNN_XXX_13.h5 ' in this repo are the pre-trained weights for each mathematical operation type (Conv, FC, Bias).


## Experiments

Training without WNN on CIFAR10:

```
python CIFAR10_without_WNN.py
```


Training with WNN on CIFAR10:

```
python CIFAR10_with_WNN.py
```

