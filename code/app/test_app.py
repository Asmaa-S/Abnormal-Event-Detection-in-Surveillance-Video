def video_to_clips(X_test,t):
    """ Video to a set of clips of length t each
    """
    import numpy as np
    sz = X_test.shape[0]-t+1
    X_test = np.expand_dims(X_test, axis=-1)
    sequences = np.zeros((sz, 10, 227, 227, 1))
    for i in range(0, sz):
        clip = np.zeros((t, 227, 227, 1))
        for j in range(0, t):
            clip[j] = X_test[i + j, :, :, :]
        sequences[i] = clip
    return np.array(sequences),sz

def t_predict_video (model, X_test, t =4):
    """ Predict on whole video
    """
    import numpy as np
    sequences,sz = video_to_clips(X_test,t)
    reconstructed_sequences = model.predict(sequences)
    sa = np.array([np.linalg.norm(np.subtract(np.squeeze(sequences[i]),np.squeeze(reconstructed_sequences[i]))) for i in range(0,sz)])
    sa_normalized = (sa - np.min(sa)) / (np.max(sa)-np.min(sa))
    sr = 1.0 - sa_normalized
    return sr, sr, sz

def test(test_video, dataset):
    """ Test the model's performance
        Plot reconstruction errors/regularity scores plots
    """
    import numpy as np
    from tensorflow.keras.models import load_model
    import os
    import h5py
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    from evaluate import plot_regularity_score, plot_reconstruction_error,calc_auc_overall
    from PIL import Image
    import re
    t = 10 #time length of each clip

    #fetching paths to test_data, job_folder and trained model
    test_dir = os.path.join('./data/{0}/testing_numpy'.format(dataset))
    
    model_folder = os.path.join('./data/{0}'.format(dataset))
    if dataset == 'UCSDped1':
        epoch = 50
        val_loss = -0.117306
    elif dataset == 'UCSDped2':
        epoch = 50
        val_loss = -0.049951
    model_filename = 'model_snapshot_e{:03d}_{:.6f}.h5'.format(epoch, val_loss)
    #load model
    temporal_model = load_model(os.path.join(model_folder, model_filename))

    #loop on all videos in the test data
    test_video = test_video.replace('.mp4','.npy')
    print(test_video)
    filepath = os.path.join(test_dir, test_video)
    X_test = np.load(filepath)
    #calculate regularity_score, reconstruction_error
    score_vid, recon_error, sz = t_predict_video(temporal_model, X_test, t)

    return np.array(score_vid)