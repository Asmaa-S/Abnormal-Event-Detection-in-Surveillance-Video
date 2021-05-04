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
    input_vol = np.zeros((X_count-t+1, t, 227, 227, 1)).astype('float64')
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

def test(logger, dataset, t, job_uuid, epoch, val_loss, visualize_score=True, visualize_frame=False,
         video_root_path):
    import numpy as np
    from keras.models import load_model
    import os
    import h5py
    from keras.utils.io_utils import HDF5Matrix
    import matplotlib.pyplot as plt
    from scipy.misc import imresize, toimage

    #fetching paths to test_data, job_folder and trained model
    test_dir = os.path.join(video_root_path, '{0}/testing_h5_t{1}'.format(dataset, t))
    job_folder = os.path.join(video_root_path,dataset,'logs/{}/jobs'.format(dataset), job_uuid)
    model_filename = 'model_snapshot_e{:03d}_{:.6f}.h5'.format(epoch, val_loss)

    #load model
    temporal_model = load_model(os.path.join(job_folder, model_filename))

    #where to save results?
    save_path = os.path.join(job_folder, 'result', str(epoch))
    os.makedirs(save_path, exist_ok=True)
    
    
