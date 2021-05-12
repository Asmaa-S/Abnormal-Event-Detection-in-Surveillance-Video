import logging
import datetime
import os
import sys
import coloredlogs
from classifier import test


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        logger.warning("Ctrl + C triggered by user, testing ended prematurely")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


###################################################################################################

#Specify Testing Parameters
video_root_path='/content/drive/MyDrive/Grad Project/data/UCSD'
dataset = 'UCSDped1' 

#these parameters identify which snapshot of the model you care to test
job_uuid = 'bc5394ac-c77d-407a-89da-b5470248f118' 
epoch = 11 
val_loss = 0.006428
time_length = 10
test_data = [epoch, val_loss, job_uuid, time_length]

#path to job folder and logs
job_folder = os.path.join(video_root_path,dataset,'logs/jobs'.format(dataset), job_uuid)
log_path = os.path.join(job_folder, 'logs')
os.makedirs(log_path, exist_ok=True)

#logs for testing
logging.basicConfig(filename=os.path.join(log_path, "test-{}.log".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))),
                    level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

#handling exceptions
sys.excepthook = handle_exception
coloredlogs.install()

#which device to use
device = 'gpu0'
if device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    logger.debug("Using CPU only")
elif device == 'gpu0':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    logger.debug("Using GPU 0")
elif device == 'gpu1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    logger.debug("Using GPU 1")
elif device == 'gpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    logger.debug("Using GPU 0 and 1")

#num of videos in the test data
n_videos = len(os.listdir(os.path.join(video_root_path, '{}/testing_frames'.format(dataset))))

#start testing
test(logger, dataset, time_length, job_uuid, epoch, val_loss, video_root_path, n_videos)
logger.info("Job {} ({}) has finished testing.".format(job_uuid, dataset))