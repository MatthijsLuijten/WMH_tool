# Stop displaying warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import utils
from plot import *
from model import *
from parameters import *
from dataloader import load_data
from preprocess import preprocess

if __name__ == '__main__':

    # Either load and preprocess data OR load datasets if they are preprocessed already
    do_preprocess = True
    if do_preprocess:
        # Load data (case numbers) and shuffle
        print('--> Loading cases')
        train_cases, test_cases = load_data()
        # cases = load_data()
