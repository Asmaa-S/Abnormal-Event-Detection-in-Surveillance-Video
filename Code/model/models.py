from keras.models import Model
from keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import ConvLSTM2D
from keras.layers import BatchNormalization, LayerNormalization
from keras.layers import TimeDistributed
from keras.layers.core import Activation
from keras.layers import Input, Flatten
import yaml

"""
    Here, we define the model's architecture.
"""


def conv_lstm_ae():
    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    t = cfg['time_length']
    input_tensor = Input(shape=(t, 227, 227, 1))

    conv1 = TimeDistributed(Conv2D(filters=128, kernel_size=11, strides=4, padding='valid', activation='elu',
                                   input_shape=(t, 227, 227, 1)))(input_tensor)
    conv1 = TimeDistributed(BatchNormalization())(conv1)

    conv2 = TimeDistributed(Conv2D(filters=64, kernel_size=5, strides=2, padding='valid', activation='elu', ))(conv1)
    conv2 = TimeDistributed(BatchNormalization())(conv2)

    conv_lstm1 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True)(conv2)
    conv_lstm1 = LayerNormalization()(conv_lstm1)
    conv_lstm2 = ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True)(conv_lstm1)
    conv_lstm2 = LayerNormalization()(conv_lstm2)
    conv_lstm3 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True)(conv_lstm2)
    conv_lstm3 = LayerNormalization()(conv_lstm3)

    de_conv = TimeDistributed(
        Conv2DTranspose(128, kernel_size=(5, 5), padding='valid', activation='elu', strides=(2, 2)))(conv_lstm3)
    de_conv = TimeDistributed(BatchNormalization())(de_conv)

    decoded = TimeDistributed(Conv2DTranspose(1, kernel_size=(11, 11), activation='elu',
                                              padding='valid', strides=(4, 4)))(de_conv)

    return Model(inputs=input_tensor, outputs=decoded)


def conv_lstm_ae_2():  # AUC 58

    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    t = cfg['time_length']
    input_tensor = Input(shape=(t, 227, 227, 1))

    conv1 = TimeDistributed(Conv2D(filters=128, kernel_size=11, strides=4, padding='valid',
                                   input_shape=(t, 227, 227, 1)))(input_tensor)
    conv1 = TimeDistributed(BatchNormalization())(conv1)
    conv1 = TimeDistributed(Activation('relu'))(conv1)

    conv2 = TimeDistributed(Conv2D(filters=64, kernel_size=5, strides=2, padding='valid'))(conv1)
    conv2 = TimeDistributed(BatchNormalization())(conv2)
    conv2 = TimeDistributed(Activation('relu'))(conv2)

    conv_lstm1 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', dropout=0.4, recurrent_dropout=0.3,
                            return_sequences=True, name='conv_lstm1', kernel_regularizer='l2')(conv2)
    conv_lstm2 = ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=0.3,
                            name='conv_lstm2', kernel_regularizer='l2')(conv_lstm1)
    conv_lstm3 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True, dropout=0.5,
                            name='conv_lstm3', kernel_regularizer='l2')(conv_lstm2)

    de_conv1 = TimeDistributed(
        Conv2DTranspose(128, kernel_size=(5, 5), padding='valid', strides=(2, 2), name='de_conv1'))(conv_lstm3)
    de_conv1 = TimeDistributed(BatchNormalization())(de_conv1)
    de_conv1 = TimeDistributed(Activation('relu'))(de_conv1)

    decoded = TimeDistributed(
        Conv2DTranspose(1, kernel_size=(11, 11), padding='valid', strides=(4, 4), name='deconv2'))(
        de_conv1)

    return Model(inputs=input_tensor, outputs=decoded)
