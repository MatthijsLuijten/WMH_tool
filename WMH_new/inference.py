# Stop displaying warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Import necessary libraries
import numpy as np
import nibabel as nib
from tqdm import tqdm

# Import custom modules
import parameters
from model import build_unet
from preprocess import preprocess
from dataloader import load_all_data
import plot


if __name__ == '__main__':

    # Load data
    print('--> Loading cases')
    cases = load_all_data()

    for c in tqdm(cases):
        # Load all MRI data for the current patient
        t1, fl, lbl, affine = preprocess(c)
        
        # Concatenate T1 and FL scans along the third dimension to create a 3D input tensor
        input = np.array([np.dstack((t1[:,:,i], fl[:,:,i])) for i in range(len(t1[0][0]))])
        
        # Initialize array to store predictions from each model
        preds = np.zeros((182, 182, 218, 1))

        # Load trained U-Net models
        models = []
        for i in range(parameters.training_ensemble):
            model = build_unet(parameters.unet_input_shape)
            model.load_weights(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(i)).replace("\\","/")).expect_partial()
            models.append(model)

        for model in models:
            # make predictions
            model_predictions = model.predict(input, batch_size=None)
        
            # update initial predictions with new predictions
            preds += model_predictions
            
            
        preds = np.divide(preds, parameters.training_ensemble)
        
        preds_thresholded = np.where(preds >= 0.35, 1, 0)
        preds_thresholded = np.transpose(preds_thresholded, (1,2,0,3))
        # plot.plot_lbl_and_pred(fl, lbl, preds_thresholded)

        # Make it a 3d nifti file
        wmh_nii = nib.Nifti1Image(preds_thresholded, affine)

        if c.startswith('2015_RUNDMC'):
            path_patient = os.path.join(parameters.path_data_2015, c).replace("\\","/")
        else:
            path_patient = os.path.join(parameters.path_data_2011, c).replace("\\","/")

        # Save the NIFTI image to a file
        nib.save(wmh_nii, os.path.join(path_patient, 'wmh_pred.nii.gz'))

