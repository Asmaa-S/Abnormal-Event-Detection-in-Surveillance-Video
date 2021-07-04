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
def get_gt_vid(video_root_path,dataset, vid_idx, pred_vid):
    """ Get a video representation for the ground truth
    """
    import numpy as np
    
    if dataset == 'UCSDped1':
        gt_data = 'UCSD_ped1'
    elif dataset == 'UCSDped2':
        gt_data = 'UCSD_ped2'
    
    gt_vid_raw = np.loadtxt('{0}/{1}/gt_files/gt_{2}_vid{3:02d}.txt'.format(video_root_path, dataset,gt_data, vid_idx+1))
    gt_vid = np.zeros_like(pred_vid)

    start = int(gt_vid_raw[0])
    end = int(gt_vid_raw[1])
    gt_vid[start:end] = 1

    return gt_vid

def regularity_score(x1, x2):
    """ Calculate a regularity score
    """
    import numpy as np
    from skimage import measure
    similarity_index = measure.compare_ssim(x1[0], x2[0], multichannel =True)
    sr = 1.0 - similarity_index
    return sr

def t_predict_video (model, X_test, t =4):
    """ Predict on whole video
    """
    import numpy as np
    reconstructed_bunch = model.predict(X_test)
    sz = X_test.shape[0]
    sa = np.array([np.linalg.norm(np.subtract(np.squeeze(X_test[i]),np.squeeze(reconstructed_bunch[i]))) for i in range(0,sz)])
    sa_normalized = (sa - min(sa)) / max(sa)
    sr = 1.0 - sa_normalized
    return sr, sa, sz
    
def t_predict_volumes(model, X_test, t =4, predict_frames = False):
    """ Predict on volumes
    """
    import numpy as np
    video_scores = []
    for number,bunch in enumerate(X_test):
        n_bunch=np.expand_dims(bunch,axis=0)
        reconstructed_bunch = model.predict(n_bunch)
        score= regularity_score(n_bunch,reconstructed_bunch)
        video_scores.append(score)
    return video_scores

def test(logger, dataset, t, job_uuid, epoch, val_loss, video_root_path, n_videos):
    """ Test the model's performance
        Plot reconstruction errors/regularity scores plots
        Plot the overall AUC
    """
    import numpy as np
    from keras.models import load_model
    import os
    import h5py
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as plt
    from evaluate import plot_regularity_score, plot_reconstruction_error,calc_auc_overall

    #fetching paths to test_data, job_folder and trained model
    test_dir = os.path.join(video_root_path, '{0}/testing_h5_t{1}'.format(dataset, t))
    job_folder = os.path.join(video_root_path,dataset,'logs/jobs',job_uuid)
    model_filename = 'model_snapshot_e{:03d}_{:.6f}.h5'.format(epoch, val_loss)

    #load model
    temporal_model = load_model(os.path.join(job_folder, model_filename))


    all_gt = []
    all_pred = []

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

        #calculate regularity_score, reconstruction_error
        score_vid, recon_error, sz = t_predict_video(temporal_model, X_test, t)
        plot_reconstruction_error(video_root_path, dataset, videoid, logger, recon_error)
        plot_regularity_score(video_root_path, dataset, videoid, logger, score_vid)
        
        #for AUC
        pred_vid = imresize(np.expand_dims(recon_error,1), (sz+t,1))
        pred_vid = np.squeeze(raw_costs)
        gt_vid = get_gt_vid(video_root_path, dataset, videoid, pred_vid)
        all_gt.append(gt_vid)
        all_pred.append(pred_vid)
        
        f.close()
    
    #calculate AUC
    calc_auc_overall(logger, video_root_path, dataset, all_gt, all_pred)
        

