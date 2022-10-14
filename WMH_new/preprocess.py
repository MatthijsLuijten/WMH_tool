import os
import parameters
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import tensorflow as tf

from plot import *

### Original image ratios
### T1 (176,256,#slices)
### FL & LBL (384,512,46)

def preprocess(c):
	# print('--> Case', c)
	if c.startswith('2015_RUNDMC'):
		path_patient = os.path.join(parameters.path_data_2015, c).replace("\\","/")
	else:
		path_patient = os.path.join(parameters.path_data_2011, c).replace("\\","/")

	# Original files
	atl_orig = nib.load(os.path.join(parameters.atlasPath, 'MNI152_T1_2mm.nii.gz'))
	t1_orig = nib.load(os.path.join(path_patient, parameters.files[1]).replace("\\","/"))
	fl_orig = nib.load(os.path.join(path_patient, parameters.files[0]).replace("\\","/"))
	lbl_orig = nib.load(os.path.join(path_patient, parameters.label).replace("\\","/"))
	
	# Nifti --> np array
	atl_img = atl_orig.get_fdata()
	t1_img = t1_orig.get_fdata()
	depth = len(t1_img[0][0])
	# t1_img = np.transpose(nib.load(T1OrigPath).get_fdata(), axes=[2,0,1])
	fl_img = fl_orig.get_fdata()
	lbl_img = lbl_orig.get_fdata()
	
	# Normalize to 0-1
	atl_norm = normalize_image(atl_img)
	t1_norm = normalize_image(t1_img)
	fl_norm = normalize_image(fl_img)
	lbl_norm = normalize_image(lbl_img)

	# Rescale image
	t1_rescaled = tf.image.resize(t1_norm, (256,256))
	fl_rescaled = tf.image.resize(fl_norm, (256,256))
	lbl_rescaled = tf.image.resize(lbl_norm, (256,256))
	
	# Print image
	# plot_image(t1_rescaled[:,:,120], f'T1 of {c}')
	# plot_image(fl_rescaled[:,:,22], f'FL of {c}')
	# plot_image(lbl_rescaled[:,:,22], f'LBL of {c}')

	return t1_rescaled[:,:,int(0.6*depth)].numpy(), fl_rescaled[:,:,13].numpy(), lbl_rescaled[:,:,13].numpy()


def normalize_image(image):
	min_val, max_val = np.min(image), np.max(image)
	range = max_val - min_val
	return (image - min_val) / range