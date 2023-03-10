# Stop displaying warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tqdm import tqdm

from plot import *
from utils import * 
from model import *
from parameters import *
from dataloader import load_pm_data
from preprocess import preprocess_pm_wmh

if __name__ == '__main__':

    # Load data (case numbers)
    print('--> Loading cases')
    cases = load_pm_data(parameters.path_pm_wmh_cases_new)
    print('--> Preprocessing cases')
    for c in tqdm(cases):
        preprocess_pm_wmh(c, parameters.path_pm_wmh_data_new)
        