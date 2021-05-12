
def compile_model(model, loss, optimizer):
    """
        Compiles the given model with a specific loss and optimizer
    """
    from keras import optimizers
    model.summary()
    if optimizer == 'sgd':
        opt = optimizers.SGD(nesterov=True)
    else:
        opt = optimizer
    model.compile(loss=loss, optimizer=opt)

def get_model_by_config(model_cfg_name):
    '''
        Get the model specified in the config file from models.py
    '''
    module = __import__('models')
    get_model_func  = getattr(module, model_cfg_name)
    return get_model_func()

def train(dataset, job_folder, logger, video_root_path):
    """
        Build and train the model
    """
    import yaml
    import os
    import numpy as np
    from preprocessing import preprocess_data
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from custom_callback import LossHistory
    import matplotlib.pyplot as plt
    from keras.utils.io_utils import HDF5Matrix

    logger.debug("Loading configs from {}".format(os.path.join(job_folder, 'config.yml')))
    with open(os.path.join(job_folder, 'config.yml'), 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    #get the parameters from the config file
    nb_epoch = cfg['epochs']
    batch_size = cfg['batch_size']
    loss = cfg['cost']
    optimizer = cfg['optimizer']
    time_length = cfg['time_length']

    #get the model
    model = get_model_by_config(cfg['model'])
    
    for layer in model.layers:
        print(layer.output_shape)

    logger.info("Compiling model with {} and {} optimizer".format(loss, optimizer))
    compile_model(model, loss, optimizer)

    logger.info("Saving model configuration to {}".format(os.path.join(job_folder, 'model.yml')))
    yaml_string = model.to_yaml()
    with open(os.path.join(job_folder, 'model.yml'), 'w') as outfile:
        yaml.dump(yaml_string, outfile)

    logger.info("Preparing training and testing data")
    preprocess_data(logger, dataset, time_length, video_root_path)
    if time_length <= 0:
        data = np.load(os.path.join(video_root_path, '{0}/training_frames_t0.npy'.format(dataset)))
        data = np.reshape(data, (len(data), 227,227,time_length, 1))
    else:
        data = np.array(HDF5Matrix(os.path.join(video_root_path, '{0}/{0}_train_t{1}.h5'.format(dataset, time_length)), 'data'))
        data = np.reshape(data, (len(data), 227,227,time_length, 1))

    snapshot = ModelCheckpoint(os.path.join(job_folder,
               'model_snapshot_e{epoch:03d}_{val_loss:.6f}.h5'))

    earlystop = EarlyStopping(patience=10)

    history_log = LossHistory(job_folder, logger)

    logger.info("Initializing training...")

    history = model.fit(
        data, data,
        batch_size=batch_size,
        epochs=nb_epoch,
        validation_split=0.2,
        shuffle='batch',
        callbacks=[snapshot, earlystop, history_log]
    )

    logger.info("Training completed!")
    np.save(os.path.join(job_folder, 'train_profile.npy'), history.history)

    n_epoch = len(history.history['loss'])
    logger.info("Plotting training profile for {} epochs".format(n_epoch))
    plt.plot(range(1, n_epoch+1),
             history.history['val_loss'],
             'g-',
             label='Val Loss')
    plt.plot(range(1, n_epoch+1),
             history.history['loss'],
             'g--',
             label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(job_folder, 'train_val_loss.png'))
