import tensorflow as tf
from tensorflow.keras import datasets, models, Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout,Input, Lambda, Reshape, Conv2DTranspose, GlobalAvgPool2D
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        return tf.random.normal(tf.shape(log_var)) * tf.exp(log_var / 2) + mean

codings_size = 2

inputs2 = Input(shape=(32,32,3), name='encoder_input')
x = Conv2D(32, 3, padding='same', activation='relu')(inputs2)
x = Conv2D(64, 3, padding='same', activation='relu',strides=(2, 2))(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)

conv_shape2=tf.shape(x)

x = Flatten()(x)
x = Dense(150, activation="relu")(x)
x = Dense(100, activation="relu")(x)
codings_mean2 = Dense(codings_size)(x)  # μ
codings_log_var2 =Dense(codings_size)(x)  # γ
codings2 = Sampling()([codings_mean2, codings_log_var2]) # z
variational_encoder2 = models.Model(
    inputs=[inputs2], outputs=[codings_mean2, codings_log_var2, codings2])

decoder_inputs2 = Input(shape=(codings_size,))
x = Dense(16*16*64, activation='relu')(decoder_inputs2)
x = tf.keras.layers.Reshape((16,16,64))(x)
x=Conv2DTranspose(32,3, activation='relu', padding='same', strides=(2,2))(x)
x=Conv2DTranspose( 3 ,3, activation='sigmoid', padding='same' )(x)
variational_decoder2 = tf.keras.Model(inputs=[decoder_inputs2], outputs=x)


_, _, codings2 = variational_encoder2(inputs2)
reconstructions2 = variational_decoder2(codings2)
variational_ae2 = tf.keras.Model(inputs=[inputs2], outputs=[reconstructions2]) # Combining encoder and decoder to form a VAE

latent_loss2 = -0.5 * tf.reduce_sum(
    1 + codings_log_var2 - tf.exp(codings_log_var2) - tf.square(codings_mean2),
    axis=-1)
variational_ae2.add_loss(tf.reduce_mean(latent_loss2) /(32*32*3)) #  we divide the result by 32*32*3 to ensure it has the appropriate scale compared to the reconstruction loss

variational_ae2.compile(loss="mse", optimizer="nadam")
history = variational_ae2.fit(X_train, X_train, epochs=25, batch_size=128,
                             validation_data=(X_test, X_test))
