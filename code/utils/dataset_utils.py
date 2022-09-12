import os
import skvideo.io
from skimage.transform import resize
from skimage.io import imsave


def video_to_frame(video_root_path, dataset, train_or_test):
    """ This function is used for preparing training & testing data by converting an input video to frames
        inputs:
            - dataset (str): name of the dataset. For ex., Avenue
            - train_or_test (str): "train" or "test" data
        returns: save frames to output directory which has the name v
    """
    video_path = os.path.join(video_root_path, dataset, '{}_videos'.format(train_or_test))
    frame_path = os.path.join(video_root_path, dataset, '{}_frames'.format(train_or_test))
    os.makedirs(frame_path, exist_ok=True)

    for video_file in os.listdir(video_path):
        if video_file.lower().endswith(('.avi', '.mp4')):
            print('==> ' + os.path.join(video_path, video_file))
            vid_frame_path = os.path.join(frame_path, os.path.basename(video_file).split('.')[0])
            os.makedirs(vid_frame_path, exist_ok=True)

            vid_cap = skvideo.io.vreader(os.path.join(video_path, video_file))
            for i, image in enumerate(vid_cap):
              image = resize(image, size, mode='reflect')
              imsave(os.path.join(vid_frame_path, '{:05d}.jpg'.format(i)), image)   # save frame as JPEG file


# Test
root_path = '/content/drive/MyDrive/Grad Project/data'
size = (227, 227)
# avenue dataset
video_to_frame(root_path, 'Avenue', 'training')
video_to_frame(root_path, 'Avenue', 'testing')