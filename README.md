
# Weight Nowcasting Network (WNN)

<!---Code for ["Learning to Boost Training by Periodic Nowcasting Near Future Weights"]-->

### Updates

- 03/07/2023: Project page built


### Abstract

Recent complicated problems require large-scale datasets and complex model architectures, however, it is difficult to train such large networks due to high computational issues. 
Significant efforts have been made to make the training more efficient such as momentum, learning rate scheduling, weight regularization, and meta-learning. Based on our observations on 1) high correlation between past weights and future weights, 2) conditions for beneficial weight prediction, and 3) feasibility of weight prediction, we propose a more general framework by intermittently skipping a handful of epochs by periodically forecasting near future weights, i.e., a Weight Nowcaster Network (WNN). As an add-on module, WNN predicts the future weights to make the learning process faster regardless of tasks and architectures.
Experimental results show that WNN can significantly save actual time cost for training with an additional marginal time to train WNN.
We validate the generalization capability of WNN under various tasks, and demonstrate that it works well even for unseen tasks. 


<p align="center">
  <img src="https://github.com/jjh6297/WNN/blob/main/Figs/thumbnail_landscape.png"/>
</p>

### WNN Architecture

WNN is composed of simple two-stream networks that use fully-connected
layers and an activation network. Feature vectors from those two networks are unified to a feature vector and it is passed through a
fully-connected layer. The predicted future weight parameters are obtained by adding outputs and input weight parameters.

<p align="center">
  <img src="https://github.com/jjh6297/WNN/blob/main/Figs/wnn_architecture.png"/>
</p>


### Dependency


<!-- dependencies: -->

| Library | Known Working | Known Not Working |  
| tensorflow | 2.3.0, 2.9.0 | <= 2.0 |
<!-- | tensorflow | 2.3.0, 2.4.1 | <= 2.0 | -->


### Usage

We provide a simple plug-in source code that can be added to your source code by using a callback function extending tf.keras.callbacks.Callback:
<!---WNN can be easily used as a callback function extending tf.keras.callbacks.Callback: -->
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

## Poster

![alt text](https://github.com/jjh6297/WNN/blob/main/Figs/ICML2023-poster_WNN_v1.0.png?raw=true)

## Slides

<a href="Figs/icml2023_slides.pdf">Slides for ICML2023</a>

## Citation

```
@inproceedings{jang2023learning,
  title={Learning to Boost Training by Periodic Nowcasting Near Future Weights},
  author={Jang, Jinhyeok and Yun, Woo-han and Kim, Won Hwa and Yoon, Youngwoo and Kim, Jaehong and Lee, Jaeyeon and Han, ByungOk},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023}
}
```
