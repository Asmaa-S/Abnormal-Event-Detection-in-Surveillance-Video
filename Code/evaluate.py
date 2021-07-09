def get_gt_range(video_root_path,dataset, vid_idx):
    """ Get the ground truth range for abnormal event in a video
    """
    import numpy as np
    if dataset == 'UCSDped1':
        gt_data = 'UCSD_ped1'
    elif dataset == 'UCSDped2':
        gt_data = 'UCSD_ped2'
    
    ret =  np.loadtxt('{0}/{1}/gt_files/gt_{2}_vid{3:02d}.txt'.format(video_root_path,dataset,gt_data,vid_idx+1))
    if(ret.shape.__len__() == 1):
        return [ret]
    return ret


def plot_regularity_score(video_root_path, dataset, videoid, logger, score_vid):
    """ Plot the regularity scores across all frames
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    save_path = os.path.join(video_root_path,dataset,'regularity_plots')
    os.makedirs(save_path, exist_ok=True)  
    
    logger.debug("Plotting regularity scores")
    plt.figure(figsize=(10, 3))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9])
    ax.plot(np.arange(1, len(score_vid)+1), score_vid, color='b', linewidth=2.0)
    plt.xlabel('Frame Number')
    plt.ylabel('Regularity score')
    plt.ylim(0, 1)
    plt.xlim(1, len(score_vid)+1)
    #ax.hlines(y=0.5, xmin=0, xmax=len(score_vid)+1, linewidth=1, color='r')

    vid_raw = get_gt_range(video_root_path,dataset, videoid)
    for event in vid_raw:
        plt.fill_between(np.arange(event[0], event[1]), 0, 1, facecolor='green', alpha=0.4)

    plt.savefig(os.path.join(save_path,'scores_{0}_video_{1:02d}.png'.format(dataset, videoid+1)), dpi=300)
    plt.close()

def plot_reconstruction_error(video_root_path, dataset, videoid, logger, score_vid):
    """ Plot the reconstruction error across all frames
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    save_path = os.path.join(video_root_path,dataset,'reconstruction_error_plots')
    os.makedirs(save_path, exist_ok=True)  
    
    logger.debug("Plotting reconstruction errors")
    plt.figure(figsize=(10, 3))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9])
    ax.plot(np.arange(1, len(score_vid)+1), score_vid, color='b', linewidth=2.0)
    plt.xlabel('Frame Number')
    plt.ylabel('Reconstruction Error')
    plt.ylim(0, 1)
    plt.xlim(1, len(score_vid)+1)
    #ax.hlines(y=0.5, xmin=0, xmax=len(score_vid)+1, linewidth=1, color='r')

    vid_raw = get_gt_range(video_root_path,dataset, videoid)
    for event in vid_raw:
        plt.fill_between(np.arange(event[0], event[1]), 0, 1, facecolor='green', alpha=0.4)

    plt.savefig(os.path.join(save_path,'errors_{0}_video_{1:02d}.png'.format(dataset, videoid+1)), dpi=300)
    plt.close()

def compute_eer(far, frr):
    cords = zip(far, frr)
    min_dist = 999999
    for item in cords:
        item_far, item_frr = item
        dist = abs(item_far-item_frr)
        if dist < min_dist:
            min_dist = dist
            eer = (item_far + item_frr) / 2
    return eer

def calc_auc_overall(logger, video_root_path, dataset, all_gt, all_pred):
    """ Calculate overall AUC
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve,auc
    import matplotlib.pyplot as plt
    import os
    
    all_gt = np.asarray(all_gt)
    all_pred = np.asarray(all_pred)
    all_gt = np.concatenate(all_gt).ravel()
    all_pred = np.concatenate(all_pred).ravel()

    fpr, tpr, thresholds = roc_curve(all_gt, all_pred, pos_label=0)
    auc = auc(fpr, tpr)
    #auc = roc_auc_score(all_gt, all_pred)
    #fpr, tpr, thresholds = roc_curve(all_gt, all_pred, pos_label=1)
    frr = 1 - tpr
    far = fpr
    eer = compute_eer(far, frr)

    logger.info("Dataset {}: Overall AUC = {:.2f}%, Overall EER = {:.2f}%".format(dataset, auc*100, eer*100))

    plt.plot(fpr, tpr)
    plt.plot([0,1],[1,0],'--')
    plt.xlim(0,1.01)
    plt.ylim(0,1.01)
    plt.title('{0} AUC: {1:.3f}, EER: {2:.3f}'.format(dataset, auc, eer))
    plt.savefig(os.path.join(video_root_path,dataset,'{}_auc.png'.format(dataset)))
    plt.close()

    return auc, eer