#This file is for running preprocessing only

import logging
import sys
from preprocessing import preprocess_data
import yaml

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()

#paths
video_root_path = '/content/drive/MyDrive/Grad Project/data/UCSD'
dataset = 'UCSDped1'

#time_length
with open('config.yml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)
    t = cfg['time_length']
#run preprocessing
preprocess_data(logger, dataset, t, video_root_path)