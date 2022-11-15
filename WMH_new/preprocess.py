import os
import parameters
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import tensorflow as tf
import time

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
	t1_path = os.path.join(path_patient, parameters.files[1]).replace("\\","/")
	fl_path = os.path.join(path_patient, parameters.files[0]).replace("\\","/")
	lbl_path = os.path.join(path_patient, parameters.label).replace("\\","/")

	if not os.path.exists(os.path.join(path_patient, 'warped_t1.nii.gz').replace("\\","/")):
		do_fsl(path_patient, t1_path, fl_path, lbl_path)

	# Path --> nifti image
	t1 = nib.load(os.path.join(path_patient, 'warped_t1.nii.gz').replace("\\","/"))
	fl_orig = nib.load(os.path.join(path_patient, 'warped_fl.nii.gz').replace("\\","/"))
	lbl_orig = nib.load(os.path.join(path_patient, 'warped_wmh.nii.gz').replace("\\","/"))
	
	# Nifti --> np array
	t1_img = t1.get_fdata()
	depth = len(t1_img[0][0])
	# t1_img = np.transpose(nib.load(T1OrigPath).get_fdata(), axes=[2,0,1])
	fl_img = fl_orig.get_fdata()
	lbl_img = lbl_orig.get_fdata()
	
	# Normalize to 0-1
	t1_norm = normalize_image(t1_img)
	fl_norm = normalize_image(fl_img)
	lbl_norm = normalize_image(lbl_img)
  
	# Threshold label to 0 or 1
	lbl_norm[lbl_norm > 0.1] = 1
	lbl_norm[lbl_norm <= 0.1] = 0

	# Rescale image
	t1_rescaled = tf.image.resize(t1_norm, (200,200))
	fl_rescaled = tf.image.resize(fl_norm, (200,200))
	lbl_rescaled = tf.image.resize(lbl_norm, (200,200))
	
	# Print image
	# for z in range(60,140,5):
	# 	plot_orig_and_lbl(t1_rescaled[:,:,z], fl_rescaled[:,:,z], lbl_rescaled[:,:,z])
	# plot_orig_and_lbl(t1_rescaled[:,:,49], fl_rescaled[:,:,49], lbl_rescaled[:,:,49])
	# plot_image(t1_rescaled[:,:,92], f'T1 of {c}')
	# plot_image(fl_rescaled[:,:,92], f'FL of {c}')
	# plot_image(lbl_rescaled[:,:,92], f'LBL of {c}')

	return t1_rescaled.numpy(), fl_rescaled.numpy(), lbl_rescaled.numpy()


def do_fsl(path_patient, t1_path, fl_path, lbl_path):
	startTime = time.time()
	wsl = 'wsl ~ -e sh -c'
	fsl = '/usr/local/fsl/bin/'
	# mnt = '/mnt/c/Users/Matthijs/Documents/RU/MSc_Thesis/WMH_tool/'
	mnt = '/mnt/c/Research/Matthijs/WMH_tool/'
	start = f'{wsl} "export FSLDIR=/usr/local/fsl; . ${{FSLDIR}}/etc/fslconf/fsl.sh; {fsl}'
	mni_bet = 'WMH_new/material/MNI152_T1_1mm_brain.nii.gz'
	mni_t1 = 'WMH_new/material/MNI152_T1_1mm.nii.gz'
	
	# Files to be made
	t1_2_flair = os.path.join(path_patient, 't1_2_flair.nii.gz').replace("\\","/")
	t1_2_flair_mat = os.path.join(path_patient, 't1_2_flair_mat.mat').replace("\\","/")

	t1_2_flair_brain = os.path.join(path_patient, 't1_2_flair_brain.nii.gz').replace("\\","/")
	t1_2_flair_brain_mask = os.path.join(path_patient, 't1_2_flair_brain_mask.nii.gz').replace("\\","/")
	fl_brain = os.path.join(path_patient, 'fl_brain.nii.gz').replace("\\","/")
	t1_2_std = os.path.join(path_patient, 't1_2_std.nii.gz').replace("\\","/")
	t1_2_std_mat = os.path.join(path_patient, 't1_2_std_mat.mat').replace("\\","/")
	transformation = os.path.join(path_patient, 'transformation.nii.gz').replace("\\","/")

	warped_t1 = os.path.join(path_patient, 'warped_t1.nii.gz').replace("\\","/")
	warped_fl = os.path.join(path_patient, 'warped_fl.nii.gz').replace("\\","/")
	warped_wmh = os.path.join(path_patient, 'warped_wmh.nii.gz').replace("\\","/")

	commands = list()

	# Map T1 to FLAIR
	commands.append([f'{start}flirt -in {mnt}{t1_path} -ref {mnt}{fl_path} -out {mnt}{t1_2_flair} -omat {mnt}{t1_2_flair_mat} -bins 256 -cost mutualinfo -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear"', '1/8 Original T1 to FLAIR space (FLIRT)'])
	
	# BET T1_2_FLAIR
	commands.append([f'{start}bet {mnt}{t1_2_flair} {mnt}{t1_2_flair_brain} -S -f 0.5 -g 0 -m"', '2/8 Bet t1_2_flair (BET)'])
	# BET FLAIR (use mask of above function)
	commands.append([f'{start}fslmaths {mnt}{fl_path} -mul {mnt}{t1_2_flair_brain_mask} {mnt}{fl_brain}"', '3/8 Bet flair (fslmath)'])

	# Map t1_2_flair to MNI (4 steps)
	commands.append([f'{start}flirt -in {mnt}{t1_2_flair_brain} -ref {mnt}{mni_bet} -out {mnt}{t1_2_std} -omat {mnt}{t1_2_std_mat} -cost corratio"', '4/8 Map t1_2_flair_brain to MNI (FLIRT)'])
	commands.append([f'{start}fnirt --in={mnt}{t1_2_flair} --aff={mnt}{t1_2_std_mat} --cout={mnt}{transformation} --ref={mnt}{mni_bet}"', '5/8 Map t1_2_flair to MNI (FNIRT)'])
	commands.append([f'{start}applywarp --ref={mnt}{mni_bet} --in={mnt}{t1_2_flair_brain} --warp={mnt}{transformation} --out={mnt}{warped_t1}"', '6/8 Map t1_2_flair_brain to MNI (APPLYWARP)'])
	
	# Map FL and label to MNI
	commands.append([f'{start}applywarp --ref={mnt}{mni_bet} --in={mnt}{fl_brain} --warp={mnt}{transformation} --out={mnt}{warped_fl}"', '7/8 Map FL to MNI (APPLYWARP)'])
	commands.append([f'{start}applywarp --ref={mnt}{mni_bet} --in={mnt}{lbl_path} --warp={mnt}{transformation} --out={mnt}{warped_wmh}"', '8/8 Map label to MNI (APPLYWARP)'])

	for command,description in commands:
		print(f'   --- {description} ---')
		os.system(command)

	endTime = time.time() - startTime
	print("      --- took", round(endTime/60.0,2), "mintues")

	
def normalize_image(image):
	min_val, max_val = np.min(image), np.max(image)
	range = max_val - min_val
	return (image - min_val) / range
