import os
import parameters
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def preprocess(c):
	print('--> Case', c)
	path_patient = os.path.join(parameters.path_data, c).replace("\\","/")

	# Original files
	T1OrigPath = os.path.join(path_patient, parameters.files[1]).replace("\\","/")
	FLOrigPath = os.path.join(path_patient, parameters.files[0]).replace("\\","/")
	labelOrigPath = os.path.join(path_patient, parameters.label).replace("\\","/")
	
	t1_img = nib.load(T1OrigPath).get_fdata()
	print(t1_img.shape)
	fl_img = nib.load(FLOrigPath).get_fdata()
	print(fl_img.shape)
	lbl_img = nib.load(labelOrigPath).get_fdata()
	print(lbl_img.shape)

	# Print image
	# image = t1_img[:,172,:]
	# plt.imshow(image, cmap='gray')
	# plt.show()
	return
