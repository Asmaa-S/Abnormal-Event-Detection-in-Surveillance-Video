'''
    Testing requires the following steps:
        1- predicting a volume and calculating the mean square error between the
            original & predicted frame --> e(t)
        2- calculating the anomaly score as follows:
            sa = [e(t) - min(e(t))]/max(e(t))
        3- regularity score:
            sr = 1-sa
        4- compare score to a threshold
'''
def t_predict (model, X, t =4):
    import numpy as np
    #get volumes 
    X_count = X.shape[0]
    input_vol = np.zeros((X_count-t+1, 227, 227, t, 1)).astype('float64')
    for i in range(X_count-t+1):
        input_vol[i] = X[i:i + t]
    #predict
    predicted_vol = model.predict(input_vol)
    #mean square error
    error_arr = np.zeros((X_count, 227, 227, 1)).astype('float64')
    for i in range(X_count-t+1):
        for j in range(t):
            error_arr[i+j] += (predicted_vol[i, j] - input_vol[i, j])**2
    return np.squeeze(error_arr)

def anomaly_score(raw_frame_cost_vid):
    score_vid = raw_frame_cost_vid - min(raw_frame_cost_vid)
    score_vid = score_vid / max(score_vid)
    return score_vid

def test(logger, dataset, t, job_uuid, epoch, val_loss, video_root_path, n_videos):
    import numpy as np
    from keras.models import load_model
    import os
    import h5py
    from keras.utils.io_utils import HDF5Matrix
    import matplotlib.pyplot as plt
    from scipy.misc import imresize, toimage
    import matplotlib.pyplot as plt

    #fetching paths to test_data, job_folder and trained model
    test_dir = os.path.join(video_root_path, '{0}/testing_h5_t{1}'.format(dataset, t))
    job_folder = os.path.join(video_root_path,dataset,'logs/jobs',job_uuid)
    model_filename = 'model_snapshot_e{:03d}_{:.6f}.h5'.format(epoch, val_loss)

    #load model
    temporal_model = load_model(os.path.join(job_folder, model_filename))

    #where to save results?
    save_path = os.path.join(job_folder, 'result', str(epoch))
    os.makedirs(save_path, exist_ok=True)

    sr_test = []
    #loop on all videos in the test data
    for videoid in range(n_videos):
        videoname = '{0}_{1:02d}.h5'.format(dataset, videoid+1)
        filepath = os.path.join(test_dir, videoname)
        logger.info("==> {}".format(filepath))

        if t > 0:
            f = h5py.File(filepath, 'r')
            filesize = f['data'].shape[0]
            f.close()
        
        #load data
        if t > 0: #if there was a time_length for the volumes
            X_test = HDF5Matrix(filepath, 'data')
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (len(X_test), 227,227,t, 1))
        else:
            X_test = np.load(os.path.join(video_root_path, '{0}/testing_numpy/testing_frames_{1:03d}.npy'.format(dataset, videoid+1))).reshape(-1, 227, 227, 1)

        #calculate errors
        et = t_predict(temporal_model, X_test, t)
        sa = anomaly_score(et)
        sr = 1-sa
        sr_test.append(sr)
        
        #####Next: sr is a matrix? how should I plot the error to know which one is abnormal?
    plt.plot(sr_test)
    plt.ylabel('regularity score Sr(t)')
    plt.xlabel('frame t')
    plt.show()


