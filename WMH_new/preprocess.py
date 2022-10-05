import os
import parameters
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import tensorflow as tf

from plot import *

### Original image ratios
### T1 (176,256,256)
### FL & LBL (384,512,23)

def preprocess(c):
	# print('--> Case', c)
	path_patient = os.path.join(parameters.path_data, c).replace("\\","/")

	# Original files
	T1OrigPath = os.path.join(path_patient, parameters.files[1]).replace("\\","/")
	FLOrigPath = os.path.join(path_patient, parameters.files[0]).replace("\\","/")
	labelOrigPath = os.path.join(path_patient, parameters.label).replace("\\","/")
	
	# Nifti --> np array
	t1_img = np.transpose(nib.load(T1OrigPath).get_fdata(), axes=[2,0,1])
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
	# plot_image(t1_rescaled[:,:,172], f'T1 of {c}')
	# plot_image(fl_rescaled[:,:,13], f'FL of {c}')
	# plot_image(lbl_rescaled[:,:,13], f'LBL of {c}')

	return t1_rescaled[:,:,172].numpy(), fl_rescaled[:,:,13].numpy(), lbl_rescaled[:,:,13].numpy()


def normalize(img):
	return img/np.max(img)