import os
import parameters
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import tensorflow as tf

### Original image ratios
### T1 (176,256,256)
### FL & LBL (384,512)

def preprocess(c):
	print('--> Case', c)
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
	image = t1_rescaled[:,:,172]
	plt.imshow(image, cmap='gray')
	plt.title('T1')
	plt.show()
	image = fl_rescaled[:,:,13]
	plt.imshow(image, cmap='gray')
	plt.title('FL')
	plt.show()
	image = lbl_rescaled[:,:,13]
	plt.imshow(image, cmap='gray')
	plt.title('Label')
	plt.show()
	return


def normalize(img):
	return img/np.max(img)