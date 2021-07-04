def get_gt_range(video_root_path,dataset, vid_idx):
    """ Get the ground truth range for abnormal event in a video
    """
    import numpy as np
    ret =  np.loadtxt('{0}/{1}/gt_files/gt_{1}_vid{2:02d}.txt'.format(video_root_path,dataset, vid_idx+1))
    if(ret.shape.__len__() == 1):
        return [ret]
    return ret


def plot_regularity_score(video_root_path, dataset, videoid, logger, score_vid):

    save_path = os.join(video_root_path,dataset,'plots')
    os.makedirs(save_path, exist_ok=True)  
    
    logger.debug("Plotting regularity scores")
    plt.figure(figsize=(10, 3))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height*0.1, box.width, box.height*0.9])
    ax.plot(np.arange(1, len(score_vid)+1), score_vid, color='b', linewidth=2.0)
    plt.xlabel('Bunch Number')
    plt.ylabel('Regularity score')
    plt.ylim(0, 1)
    plt.xlim(1, len(score_vid)+1)

    vid_raw = get_gt_range(video_root_path,dataset, videoid)
    for event in vid_raw:
        plt.fill_between(np.arange(event[0], event[1]), 0, 1, facecolor='red', alpha=0.4)

    plt.savefig(os.path.join(save_path, 'scores','scores_{0}_video_{1:02d}.png'.format(dataset, videoid+1)), dpi=300)
    plt.close()

