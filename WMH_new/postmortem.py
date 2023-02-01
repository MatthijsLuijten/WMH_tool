# Stop displaying warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
from tqdm import tqdm

from plot import *
from utils import * 
from model import *
from parameters import *
from dataloader import load_pm_data
from preprocess import preprocess_pm_mri, preprocess_pm_wmh

if __name__ == '__main__':

    path = parameters.path_pm_wmh_cases

    # Either load and preprocess data OR load datasets if they are preprocessed already
    do_preprocess = True
    if do_preprocess:
        # Load data (case numbers)
        print('--> Loading cases')
        cases = load_pm_data(path)

        if path == parameters.path_pm_mri_cases:
            data_path = parameters.path_pm_mri_data
            pm_data = []
            print('--> Preprocessing cases')
            t1_cmap, fl_cmap = get_cmap()
            for c in tqdm(cases):
                t1, fl = preprocess_pm_mri(c, t1_cmap, fl_cmap, data_path)
                pm_data.append(np.dstack((t1,fl)))
            
            pm_data = np.array(pm_data) 
            # np.save(path_pm_mri_dataset, pm_data)
        else:
            data_path = parameters.path_pm_wmh_data
            pm_data = []
            print('--> Preprocessing cases')
            t1_cmap, fl_cmap = get_cmap()
            for c in tqdm(cases):
                t1, fl = preprocess_pm_wmh(c, t1_cmap, fl_cmap, data_path)
                pm_data.append(np.dstack((t1,fl)))
            
            pm_data = np.array(pm_data) 
            # np.save(path_pm_wmh_dataset, pm_data)
        
        
    else:
        if path == parameters.path_pm_mri_cases:
            pm_data = np.load(path_pm_mri_dataset)
        else:
            pm_data = np.load(path_pm_wmh_dataset)

    # for d in pm_data:
    #     plot_pm_data(d[:,:,0], d[:,:,1])


    model = build_unet(unet_input_shape)
    temp_pred = model.predict(np.array([pm_data[0]]), batch_size=None)[0]
    temp_pred = np.ndarray(np.shape(temp_pred))
    preds = []
    for i in range(np.shape(pm_data)[0]):
        preds.append(temp_pred)

    for ensemble in range(training_ensemble):
        model = build_unet(unet_input_shape)
        model.load_weights(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/")).expect_partial()
        print(f'--> Loaded model {ensemble+1} from', os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/"))

        print('   --> Make predictions')
        predictions = model.predict(pm_data, batch_size=16)
        for i, p in enumerate(preds):
            preds[i] = p + predictions[i]
    
    preds = np.divide(preds, training_ensemble)
    preds = np.where(preds > 0.35, 1., 0.)

    for i,p in enumerate(preds):
        plot_pm_prediction(pm_data[i][:,:,0], pm_data[i][:,:,1], p, i)