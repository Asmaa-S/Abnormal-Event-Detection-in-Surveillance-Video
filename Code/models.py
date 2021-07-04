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
    input_tensor = Input(shape=(t, 227, 227, 1))
    
    conv1 = TimeDistributed(Conv2D(filters=128, kernel_size= 11, strides= 4, padding='valid',
             input_shape=(t,227, 227, 1)))(input_tensor)
    conv1 = TimeDistributed(BatchNormalization())(conv1)
    conv1 = TimeDistributed(Activation('relu'))(conv1)

    conv2 = TimeDistributed(Conv2D(filters=64, kernel_size=5, strides=2,
     padding='valid'))(conv1)
    conv2 = TimeDistributed(BatchNormalization())(conv2)
    conv2 = TimeDistributed(Activation('relu'))(conv2)

    convlstm1 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same',  
                            return_sequences=True, name='convlstm1',kernel_regularizer='l2')(conv2)
    convlstm2 = ConvLSTM2D(32, kernel_size=(3, 3), padding='same', return_sequences=True,
                             name='convlstm2',kernel_regularizer='l2')(convlstm1)
    convlstm3 = ConvLSTM2D(64, kernel_size=(3, 3), padding='same', return_sequences=True,
                         name='convlstm3',kernel_regularizer='l2')(convlstm2)
    
    deconv1 = TimeDistributed(Conv2DTranspose(128, kernel_size=(5, 5), padding='valid', strides=(2, 2), name='deconv1'))(convlstm3)
    deconv1 = TimeDistributed(BatchNormalization())(deconv1)
    deconv1 = TimeDistributed(Activation('relu'))(deconv1)

    decoded = TimeDistributed(Conv2DTranspose(1, kernel_size=(11, 11), padding='valid', strides=(4, 4), name='deconv2'))(
        deconv1)
    
    return Model(inputs=input_tensor, outputs=decoded)