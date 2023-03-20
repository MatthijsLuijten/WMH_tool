# Stop displaying warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

import numpy as np

from plot import *
from model import *
from multi_model import *
from parameters import *
from dataloader import load_pm_data
from preprocess import preprocess_pm_data_lfb
import postprocess

if __name__ == '__main__':

    # Load and preprocess data 
    do_preprocess = True
    if do_preprocess:
        # Load data (case numbers) and shuffle
        print('--> Loading cases')
        cases = load_pm_data(path_pm_wmh_cases)

        print('--> Preprocessing training and test cases')
        input_img, lfb_img = preprocess_pm_data_lfb(cases)

        # print('--> Saving datasets')
        # save_pm_datasets(train_img, train_lbl_wmh, train_lbl_nawm, train_lbl_gm, test_img, test_lbl_wmh, test_lbl_nawm, test_lbl_gm)     

    # Make pred arrays for ensemble prediction
    preds = [np.zeros((200,200,4)) for _ in range(len(input_img))]
    
    for ensemble in range(training_ensemble):
        model = build_multi_unet(unet_input_shape)
        model.load_weights(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/")).expect_partial()
        print(f'--> Loaded model {ensemble+1} from', os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/"))

        print('--> Make predictions')
        predictions = model.predict(input_img, batch_size=16)
        for i, p in enumerate(preds):
            preds[i] += predictions[i]

    # Average and threshold prediction
    preds = tf.math.argmax(preds, axis=-1)
    preds = preds[..., tf.newaxis]

    # Postprocess (transition zones and correlation)
    preds, wmh_zones, nawm_zones, gm_mask, lfb_img = postprocess.run(input_img, preds, lfb_img, cases)
    # for i in range(len(preds)):
    #     plot_prediction(input_img[i][:,:,1], wmh_zones[i], nawm_zones[i], gm_mask[i], lfb_img[i], i)
