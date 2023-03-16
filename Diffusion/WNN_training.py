
import os 
from tensorflow.keras.layers import Concatenate, MaxPooling2D, Lambda, ZeroPadding2D, BatchNormalization, Activation, AveragePooling2D, Reshape, concatenate, DepthwiseConv2D, SpatialDropout2D, LeakyReLU
from tensorflow.keras.layers import Concatenate, MaxPooling2D, Lambda, ZeroPadding2D, BatchNormalization, Activation, AveragePooling2D, Reshape, concatenate, DepthwiseConv2D, SpatialDropout2D, LeakyReLU
from tensorflow.keras.layers import Input, Dense, Permute, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, AveragePooling2D, GlobalMaxPooling2D, LeakyReLU, Reshape, Concatenate
from tensorflow.keras.layers import Activation, Dropout, Flatten, concatenate, Maximum, Lambda, Multiply, Add, add, multiply, SpatialDropout2D, GaussianDropout, AlphaDropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, Adadelta, RMSprop
from TrActLayer import trAct_1D_Exp, trAct_2D_Exp
import math
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
from keras import layers
import numpy as np
from WNN import *


global model2_Conv
global model2_FC
global model2_B
global prediction

dataset_name = "oxford_flowers102"
dataset_repetitions = 5
num_epochs = 50 #50  # train for at least 50 epochs for good results
image_size = 64
# KID = Kernel Inception Distance, see related section
kid_image_size = 75
kid_diffusion_steps = 5
plot_diffusion_steps = 20

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 32
embedding_max_frequency = 1000.0
widths = [32, 64, 96, 128]
block_depth = 2

# optimization
batch_size = 64
# batch_size = 8

ema = 0.999
learning_rate = 1e-3
weight_decay = 1e-4


NumLength=5
    

model2_Conv = CNNmodel()
model2_Conv.compile(loss='mae', optimizer=Adam(),metrics=['mae'])  
model2_Conv.load_weights('NWNN_Conv_13.h5.h5')
model2_FC = CNNmodel()
model2_FC.compile(loss='mae', optimizer=Adam(),metrics=['mae'])  
model2_FC.load_weights('NWNN_FC_13.h5')
model2_B = CNNmodel()
model2_B.compile(loss='mae', optimizer=Adam(),metrics=['mae'])  
model2_B.load_weights('NWNN_Bias_13.h5.h5')
    



def preprocess_image(data):
    # center crop image
    height = tf.shape(data["image"])[0]
    width = tf.shape(data["image"])[1]
    crop_size = tf.minimum(height, width)
    image = tf.image.crop_to_bounding_box(
        data["image"],
        (height - crop_size) // 2,
        (width - crop_size) // 2,
        crop_size,
        crop_size,
    )

    # resize and clip
    # for image downsampling it is important to turn on antialiasing
    image = tf.image.resize(image, size=[image_size, image_size], antialias=True)
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)


def prepare_dataset(split):
    # the validation dataset is shuffled as well, because data order matters
    # for the KID estimation
    return (
        tfds.load(dataset_name, split=split, shuffle_files=True)
        .map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat(dataset_repetitions)
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


train_dataset = prepare_dataset("train[:80%]+validation[:80%]+test[:80%]")
val_dataset = prepare_dataset("train[80%:]+validation[80%:]+test[80%:]")



class KID(keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)

        self.kid_tracker = keras.metrics.Mean(name="kid_tracker")
        self.encoder = keras.Sequential(
            [
                keras.Input(shape=(image_size, image_size, 3)),
                layers.Rescaling(255.0),
                layers.Resizing(height=kid_image_size, width=kid_image_size),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    input_shape=(kid_image_size, kid_image_size, 3),
                    weights="imagenet",
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, features_1, features_2):
        feature_dimensions = tf.cast(tf.shape(features_1)[1], dtype=tf.float32)
        return (features_1 @ tf.transpose(features_2) / feature_dimensions + 1.0) ** 3.0

    def update_state(self, real_images, generated_images, sample_weight=None):
        real_features = self.encoder(real_images, training=False)
        generated_features = self.encoder(generated_images, training=False)

        # compute polynomial kernels using the two sets of features
        kernel_real = self.polynomial_kernel(real_features, real_features)
        kernel_generated = self.polynomial_kernel(
            generated_features, generated_features
        )
        kernel_cross = self.polynomial_kernel(real_features, generated_features)

        # estimate the squared maximum mean discrepancy using the average kernel values
        batch_size = tf.shape(real_features)[0]
        batch_size_f = tf.cast(batch_size, dtype=tf.float32)
        mean_kernel_real = tf.reduce_sum(kernel_real * (1.0 - tf.eye(batch_size))) / (
            batch_size_f * (batch_size_f - 1.0)
        )
        mean_kernel_generated = tf.reduce_sum(
            kernel_generated * (1.0 - tf.eye(batch_size))
        ) / (batch_size_f * (batch_size_f - 1.0))
        mean_kernel_cross = tf.reduce_mean(kernel_cross)
        kid = mean_kernel_real + mean_kernel_generated - 2.0 * mean_kernel_cross

        # update the average KID estimate
        self.kid_tracker.update_state(kid)

    def result(self):
        return self.kid_tracker.result()

    def reset_state(self):
        self.kid_tracker.reset_state()


def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    )
    return embeddings


def ResidualBlock(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(
            width, kernel_size=3, padding="same", activation=keras.activations.swish
        )(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x

    return apply


def get_network(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth)([x, skips])

    x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")



class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = keras.models.clone_model(self.network)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates**2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images, image_size, image_size, 3))
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images):
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(batch_size, image_size, image_size, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=kid_diffusion_steps
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()


model = DiffusionModel(image_size, widths, block_depth)

model.compile(
    optimizer=keras.optimizers.experimental.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    ),
    loss=keras.losses.mean_absolute_error,
)

checkpoint_path = "checkpoints/diffusion_model"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_kid",
    mode="min",
    save_best_only=False,
)

model.normalizer.adapt(train_dataset)



trial=15

lcnt=0
for ll in model.layers:
    try:
        ll2 = ll.layers
        if len(ll2)>0:
            for ll2_2 in ll2:
                if 'norm' not in ll2_2.name: 
                    temp = ll2_2.get_weights()
                    for tt in temp:
                        exec('Weights'+ str(lcnt) + '= np.zeros( tt.shape + (NumLength,) )')    				
                        lcnt = lcnt+1    

    except:
        ll2_2 = ll
        if 'norm' not in ll2_2.name: 
            temp = ll2_2.get_weights()
            for tt in temp:
                exec('Weights'+ str(lcnt) + '= np.zeros( tt.shape + (NumLength,) )')                        
                lcnt = lcnt+1 
                
             
NLoss=[]
ILoss=[]
VNLoss=[]
VILoss=[]
KID=[]
cnt=0
for ep in range(num_epochs//5):
    for ii in range(5):
        hist = model.fit(
            train_dataset,
            epochs=cnt+1,initial_epoch=cnt, #steps_per_epoch=8, validation_steps=1,
            validation_data=val_dataset,
            callbacks=[checkpoint_callback,],)        
        lcnt=0
        for ll in model.layers:
            try:
                ll2 = ll.layers
                if len(ll2)>0:
                    for ll2_2 in ll2:
                        if 'norm' not in ll2_2.name: 
                            temp = ll2_2.get_weights()
                            for tt in temp:
                                exec('Weights'+ str(lcnt) + '[...,ii]= tt')    				
                                lcnt = lcnt+1    
            except:
                ll2_2 = ll
                if 'norm' not in ll2_2.name: 
                    temp = ll2_2.get_weights()
                    for tt in temp:
                        exec('Weights'+ str(lcnt) + '[...,ii]= tt')    				
                        lcnt = lcnt+1 	
        NLoss.append(hist.history['n_loss'][-1])
        ILoss.append(hist.history['i_loss'][-1])    
        VNLoss.append(hist.history['val_n_loss'][-1])
        VILoss.append(hist.history['val_i_loss'][-1])
        KID.append(hist.history['val_kid'][-1])
        cnt=cnt+1
    lcnt=0
    for ll in model.layers:
        try:
            ll2 = ll.layers
            if len(ll2)>0:
                for ll2_2 in ll2:
                    if 'norm' not in ll2_2.name: 
                        temp = ll2_2.get_weights()
                        New_Weights=temp
                        for tt_idx in range(len(temp)):
                            tt = temp[tt_idx]
                            if len(tt.shape)==4:
                                exec('prediction = get_ConvLayer_pred(Weights'+ str(lcnt) + ',model2_Conv)')
                                New_Weights[tt_idx]=prediction

                            elif len(tt.shape)==3:
                                exec('prediction = get_DepthConvLayer_predConvLayer_pred(Weights'+ str(lcnt) + ',model2_Conv)')
                                New_Weights[tt_idx]=prediction
                                
                            elif len(tt.shape)==2:
                                exec('prediction = get_FCLayer_pred(Weights'+ str(lcnt) + ',model2_FC)')
                                New_Weights[tt_idx]=prediction
                            elif len(tt.shape)==1:
                                exec('prediction = get_BiasLayer_pred(Weights'+ str(lcnt) + ',model2_B)')
                                New_Weights[tt_idx]=prediction       

                            lcnt = lcnt+1    
                        ll2_2.set_weights(New_Weights)	
        except:
            ll2_2 = ll
            if 'norm' not in ll2_2.name: 
                temp = ll2_2.get_weights()
                New_Weights=temp
                for tt_idx in range(len(temp)):
                    tt = temp[tt_idx]
                    if len(tt.shape)==4:
                        exec('prediction = get_ConvLayer_pred(Weights'+ str(lcnt) + ',model2_Conv)')
                        New_Weights[tt_idx]=prediction

                    elif len(tt.shape)==3:
                        exec('prediction = get_DepthConvLayer_predConvLayer_pred(Weights'+ str(lcnt) + ',model2_Conv)')
                        New_Weights[tt_idx]=prediction

                    elif len(tt.shape)==2:
                        exec('prediction = get_FCLayer_pred(Weights'+ str(lcnt) + ',model2_FC)')
                        New_Weights[tt_idx]=prediction
                    elif len(tt.shape)==1:
                        exec('prediction = get_BiasLayer_pred(Weights'+ str(lcnt) + ',model2_B)')
                        New_Weights.append(prediction)

                    lcnt = lcnt+1    
                ll2_2.set_weights(New_Weights)	
                        
import scipy.io as sio
sio.savemat('Save_Diffusion_Model_WNN_trial'+str(trial)+'.mat', mdict = {'NLoss':np.array(NLoss), 'ILoss':np.array(ILoss), 'VNLoss':np.array(VNLoss), 'VILoss':np.array(VILoss), 'KID':np.array(KID)})
model.save_weights('Weights_Diffusion_Model_WNN_trial'+str(trial)+'.pd')


model.plot_images()
