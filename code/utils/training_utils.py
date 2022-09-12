import yaml
from generator import train_model, plot_loss
import h5py
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def compile_model(model, loss, optimizer):
    """
        Compiles the given model with a specific loss and optimizer
    """
    from keras import optimizers
    model.summary()
    if optimizer == 'sgd':
        opt = optimizers.SGD(nesterov=True)
    elif optimizer == 'adam':
        opt = optimizers.Adam(lr=1e-4, decay=1e-4 / 100, epsilon=1e-6)
    else:
        opt = optimizer

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['cosine_similarity'])


def get_model_by_config(model_cfg_name):
    """
        Get the model specified in the config file from models.py
    """
    module = __import__('models')
    get_model_func = getattr(module, model_cfg_name)
    return get_model_func()


def train(dataset, job_folder, logger, video_root_path):
    """
        Build and train the model
    """

    logger.debug("Loading configs from {}".format(os.path.join(job_folder, 'config.yml')))
    with open(os.path.join(job_folder, 'config.yml'), 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # get the parameters from the config file
    nb_epoch = cfg['epochs']
    batch_size = cfg['batch_size']
    loss = cfg['cost']
    optimizer = cfg['optimizer']
    time_length = cfg['time_length']

    # get the model
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
    hdf5_path = os.path.join(video_root_path, '{0}/{0}_train_t{1}.h5'.format(dataset, time_length))
    with h5py.File(hdf5_path, 'r') as hf:
        sample_counts = hf['data'].shape[0]

    history = train_model(model, hdf5_path, 0.2, batch_size, sample_counts, nb_epoch, logger, job_folder)
    logger.info("Training completed!")

    plot_loss(history, job_folder, nb_epoch, logger)
