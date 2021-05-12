
def conv_lstm_ae():
    from keras.models import Model
    from keras.layers import Conv3D,ConvLSTM2D,Conv3DTranspose
    from keras.layers.convolutional import Conv2D, Conv2DTranspose
    from keras.layers.convolutional_recurrent import ConvLSTM2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.wrappers import TimeDistributed
    from keras.layers.core import Activation
    from keras.layers import Input, Flatten
    import yaml

    with open('config.yml', 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    t = cfg['time_length']
    input_tensor = Input(shape=(227, 227,t, 1))
    
    conv1 = Conv3D(filters=128, kernel_size=(11, 11, 1), strides=(4, 4,1), padding='valid', input_shape=(227, 227, t, 1)
            ,activation='relu')(input_tensor)
    conv2 = Conv3D(filters=64, kernel_size=(5, 5,1), strides=(2, 2,1), padding='valid', activation='relu')(conv1)
    
    
    convlstm1 = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', dropout=0.4,
             recurrent_dropout=0.3, return_sequences=True)(conv2)
    convlstm2 = ConvLSTM2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', dropout=0.3,
                     return_sequences=True)(convlstm1)
    convlstm3 = ConvLSTM2D(filters=64, kernel_size=(3, 3), strides=1,
                     return_sequences=True, padding='same', dropout=0.5)(convlstm2)

    conv3 = Conv3DTranspose(filters=128,kernel_size=(5,5,1),strides=(2,2,1),padding='valid',activation='relu')(convlstm3)
    decoded = Conv3DTranspose(filters=1,kernel_size=(11,11,1),strides=(4,4,1),padding='valid',activation='relu')(conv3)

    return Model(inputs=input_tensor, outputs=decoded)
