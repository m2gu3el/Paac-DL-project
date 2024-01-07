import tensorflow as tf
from tensorflow.keras import datasets, models, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout,Input, Lambda, Reshape, Conv2DTranspose, GlobalAvgPool2D
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

conv_encoder = tf.keras.Sequential([
    Conv2D(16, 3, padding="same", activation="relu", input_shape=(69, 69, 3)),
    MaxPooling2D(pool_size=2),
    Conv2D(32, 3, padding="same", activation="relu"),
    MaxPooling2D(pool_size=2),
    Conv2D(64, 3, padding="same", activation="relu"),
    MaxPooling2D(pool_size=2),
    Conv2D(30, 3, padding="same", activation="relu"),
    GlobalAvgPool2D()
])

conv_decoder = tf.keras.Sequential([
    Input(shape=(30,)),
    Dense(23*23*64, activation="relu"),
    Reshape((23, 23, 64)),
    Conv2DTranspose(64, 3, strides=3, activation="relu", padding="same"),
    Conv2DTranspose(3, 3, strides=1, activation="sigmoid", padding="same"),  # Use sigmoid for RGB images
])

# Combining Encoder and decoder to make an autoencoder
conv_ae = Sequential([conv_encoder, conv_decoder])

conv_ae.compile(optimizer='nadam', loss='mse')
