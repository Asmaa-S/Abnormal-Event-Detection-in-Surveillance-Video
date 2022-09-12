import h5py
import os
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Code.model.custom_callback import LossHistory
import matplotlib.pyplot as plt


def train_model(model, video_data_fn, validation_ratio, batch_size, sample_count, nb_epoch, logger, job_folder):
    """ Train the video classification model
    """
    snapshot = ModelCheckpoint(os.path.join(job_folder,
                                            'model_snapshot_e{epoch:03d}_{val_loss:.6f}.h5'))
    earlystop = EarlyStopping(patience=10)
    history_log = LossHistory(job_folder, logger)

    with h5py.File(video_data_fn, "r") as video_data:
        sample_idxs = range(0, sample_count)
        # sample_idxs = np.random.permutation(sample_idxs)
        training_sample_idxs = sample_idxs[0:int((1 - validation_ratio) * sample_count)]
        validation_sample_idxs = sample_idxs[int((1 - validation_ratio) * sample_count):]

        training_sequence_generator = generate_training_sequences(batch_size=batch_size,
                                                                  video_data=video_data,
                                                                  training_sample_idxs=training_sample_idxs)
        validation_sequence_generator = generate_validation_sequences(batch_size=batch_size,
                                                                      video_data=video_data,
                                                                      validation_sample_idxs=validation_sample_idxs)
        logger.info("Initializing training...")

        history = model.fit_generator(generator=training_sequence_generator,
                                      validation_data=validation_sequence_generator,
                                      steps_per_epoch=len(training_sample_idxs) // batch_size,
                                      validation_steps=len(validation_sample_idxs) // batch_size,
                                      epochs=nb_epoch,
                                      callbacks=[snapshot, earlystop, history_log])
    return history


def generate_training_sequences(batch_size, video_data, training_sample_idxs):
    """ Generates training sequences on demand
    """
    while True:
        # generate sequences for training
        training_sample_count = len(training_sample_idxs)
        batches = int(training_sample_count / batch_size)
        remainder_samples = training_sample_count % batch_size
        if remainder_samples:
            batches = batches + 1
        # generate batches of samples
        for idx in range(0, batches):
            if idx == batches - 1:
                batch_idxs = training_sample_idxs[idx * batch_size:]
            else:
                batch_idxs = training_sample_idxs[idx * batch_size:idx * batch_size + batch_size]
            batch_idxs = sorted(batch_idxs)

            data = video_data['data'][batch_idxs]

            yield (np.array(data), np.array(data))


def generate_validation_sequences(batch_size, video_data, validation_sample_idxs):
    """ Generates validation sequences on demand
    """
    while True:
        # generate sequences for validation
        validation_sample_count = len(validation_sample_idxs)
        batches = int(validation_sample_count / batch_size)
        remainder_samples = validation_sample_count % batch_size
        if remainder_samples:
            batches = batches + 1
        # generate batches of samples
        for idx in range(0, batches):
            if idx == batches - 1:
                batch_idxs = validation_sample_idxs[idx * batch_size:]
            else:
                batch_idxs = validation_sample_idxs[idx * batch_size:idx * batch_size + batch_size]
            batch_idxs = sorted(batch_idxs)

            data = video_data['data'][batch_idxs]

            yield np.array(data), np.array(data)


def plot_loss(history, job_folder, nb_epoch, logger):
    """
        Plot the loss of the validation & training
    """
    np.save(os.path.join(job_folder, 'train_profile.npy'), history.history, nb_epoch)
    n_epoch = len(history.history['loss'])
    logger.info("Plotting training profile for {} epochs".format(n_epoch))
    plt.plot(range(1, n_epoch + 1),
             history.history['val_loss'],
             'g-',
             label='Val Loss')
    plt.plot(range(1, n_epoch + 1),
             history.history['loss'],
             'g--',
             label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(job_folder, 'train_val_loss.png'))
