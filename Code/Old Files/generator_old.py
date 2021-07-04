import glob
import os
import random

import tensorflow as tf
import h5py
import numpy as np


def split_data_from_h5(h5, train_size):
    with h5py.File(h5, 'r') as hf:
        ds = hf['data']
        train = ds[0:train_size]
        val =  ds[train_size:]
    return train, val



def data_from_h5(h5):
    with h5py.File(h5, 'r') as hf:
        ds = hf['data']
    return ds


def generator(batch_size, dset):
    """A generator that returns 5 images plus a result image"""
    if isinstance(dset, str):
        dset = data_from_h5(dset)

    counter = 0
    while True:

        if (counter + batch_size >= dset.shape[0]):
            counter = 0

        input_vol = dset[counter: counter + batch_size]

        yield (input_vol, input_vol)
        counter += batch_size
