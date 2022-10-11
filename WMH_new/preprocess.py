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
	T1OrigPath = os.path.join(path_patient, parameters.files[1]).replace("\\","/")
	FLOrigPath = os.path.join(path_patient, parameters.files[0]).replace("\\","/")
	labelOrigPath = os.path.join(path_patient, parameters.label).replace("\\","/")
	
	# Nifti --> np array
	t1_img = nib.load(T1OrigPath).get_fdata()
	depth = len(t1_img[0][0])
	# t1_img = np.transpose(nib.load(T1OrigPath).get_fdata(), axes=[2,0,1])
	fl_img = nib.load(FLOrigPath).get_fdata()
	lbl_img = nib.load(labelOrigPath).get_fdata()
	
	# Normalize to 0-1
	t1_norm = tf.cast(t1_img, tf.float32) / np.max(t1_img)
	fl_norm = tf.cast(fl_img, tf.float32) / np.max(fl_img)
	lbl_norm = tf.cast(lbl_img, tf.float32) / np.max(lbl_img)

	# Rescale image
	t1_rescaled = tf.image.resize(t1_norm, (256,256))
	fl_rescaled = tf.image.resize(fl_norm, (256,256))
	lbl_rescaled = tf.image.resize(lbl_norm, (256,256))
	
	# Print image
	# plot_image(t1_rescaled[:,:,120], f'T1 of {c}')
	# plot_image(fl_rescaled[:,:,22], f'FL of {c}')
	# plot_image(lbl_rescaled[:,:,22], f'LBL of {c}')

	return t1_rescaled[:,:,int(0.6*depth)].numpy(), fl_rescaled[:,:,13].numpy(), lbl_rescaled[:,:,13].numpy()
