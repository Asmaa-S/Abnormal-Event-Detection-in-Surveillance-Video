'''
    Testing requires the following steps:
        1- predicting a volume and calculating the mean square distance between the
            original & predicted frame --> e(t)
        2- calculating the anomaly score as follows:
            sa = [e(t) - min(e(t))]/max(e(t))
        3- regularity score:
            sr = 1-sa
        4- compare score to a threshold
'''

def regularity_score(x1, x2):
    """ Calculate a regularity score
    """
    import numpy as np
    from skimage import measure
    similarity_index = measure.compare_ssim(x1[0], x2[0], multichannel =True)
    sr = 1.0 - similarity_index
    #frame_diff = np.array(np.subtract(x1, x2)) ** 2
    #sa = (frame_diff - np.min(frame_diff)) / np.max(frame_diff)
    #sr = 1.0 - abs(sa.mean())
    return sr

def score_frames(n_bunch,reconstructed_bunch):
    """ Predict score on frames of a bunch
    """
    frame_scores = []
    for i,frame in enumerate(n_bunch):
        frame_reconstructed = reconstructed_bunch[i]
        score= regularity_score(frame,frame_reconstructed)
        frame_scores.append(score)
    return frame_scores

def t_predict_volumes(model, X_test, t =4, predict_frames = False):
    """ Predict on volumes
    """
    import numpy as np
    video_scores = []
    for number,bunch in enumerate(X_test):
        n_bunch=np.expand_dims(bunch,axis=0)
        reconstructed_bunch = model.predict(n_bunch)
        if not predict_frames:
            score= regularity_score(n_bunch,reconstructed_bunch)
            video_scores.append(score)
            threshold = 0.5
            print("regularity_score = ", score)
            if score > threshold:
                print("Anomalous bunch of range {0} to {1}".format(number, number+t))
            else:
                print("Bunch Normalof range {0} to {1}".format(number, number+t))
        else:
            video_scores.append(score_frames(n_bunch,reconstructed_bunch))
        return video_scores

def test(logger, dataset, t, job_uuid, epoch, val_loss, video_root_path, n_videos):
    import numpy as np
    from keras.models import load_model
    import os
    import h5py
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    from evaluate import plot_regularity_score

    #fetching paths to test_data, job_folder and trained model
    test_dir = os.path.join(video_root_path, '{0}/testing_h5_t{1}'.format(dataset, t))
    job_folder = os.path.join(video_root_path,dataset,'logs/jobs',job_uuid)
    model_filename = 'model_snapshot_e{:03d}_{:.6f}.h5'.format(epoch, val_loss)

    #load model
    temporal_model = load_model(os.path.join(job_folder, model_filename))

    #loop on all videos in the test data
    for videoid in range(n_videos):
        videoname = '{0}_{1:02d}.h5'.format(dataset, videoid+1)
        filepath = os.path.join(test_dir, videoname)
        logger.info("==> {}".format(filepath))

        if t > 0:
            f = h5py.File(filepath, 'r')
            X_test = f['data']
            filesize = X_test.shape[0]

        
        #load data
        if t > 0: #if there was a time_length for the volumes
            X_test = np.asarray(X_test)
        else:
            X_test = np.load(os.path.join(video_root_path, '{0}/testing_numpy/testing_frames_{1:03d}.npy'.format(dataset, videoid+1)))

        #calculate errors
        score_vid = t_predict_volumes(temporal_model, X_test, t)
        plot_regularity_score(video_root_path, dataset, videoid, logger, score_vid)
        f.close()
        

