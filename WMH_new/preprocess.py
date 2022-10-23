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
	startTime = time.time()
	print('--> Case', c)
	if c.startswith('2015_RUNDMC'):
		path_patient = os.path.join(parameters.path_data_2015, c).replace("\\","/")
	else:
		path_patient = os.path.join(parameters.path_data_2011, c).replace("\\","/")

	# Original files
	t1_path = os.path.join(path_patient, parameters.files[1]).replace("\\","/")
	fl_path = os.path.join(path_patient, parameters.files[0]).replace("\\","/")
	lbl_path = os.path.join(path_patient, parameters.label).replace("\\","/")

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

	# Rescale image
	t1_rescaled = tf.image.resize(t1_norm, (200,200))
	fl_rescaled = tf.image.resize(fl_norm, (200,200))
	lbl_rescaled = tf.image.resize(lbl_norm, (200,200))
	
	# Print image
	# plot_image(t1_rescaled[:,:,92], f'T1 of {c}')
	# plot_image(fl_rescaled[:,:,92], f'FL of {c}')
	# plot_image(lbl_rescaled[:,:,92], f'LBL of {c}')

	endTime = time.time() - startTime
	print("      --- took", round(endTime/60.0,2), "mintues")

	return t1_rescaled.numpy(), fl_rescaled.numpy(), lbl_rescaled.numpy()


def do_fsl(path_patient, t1_path, fl_path, lbl_path):
	wsl = 'wsl ~ -e sh -c'
	fsl = '/usr/local/fsl/bin/'
	mnt = '/mnt/c/Users/Matthijs/Documents/RU/MSc_Thesis/WMH_tool/'
	start = f'{wsl} "export FSLDIR=/usr/local/fsl; . ${{FSLDIR}}/etc/fslconf/fsl.sh; {fsl}'
	mni_bet = 'WMH_new/material/MNI152_T1_1mm_brain.nii.gz'
	mni_t1 = 'WMH_new/material/MNI152_T1_1mm.nii.gz'
	
	# Files to be made
	bet_t1 = os.path.join(path_patient, 'bet_t1.nii.gz').replace("\\","/")
	bet_fl = os.path.join(path_patient, 'bet_fl.nii.gz').replace("\\","/")

	t1_2_flair = os.path.join(path_patient, 't1_2_flair.nii.gz').replace("\\","/")
	t1_2_flair_mat = os.path.join(path_patient, 't1_2_flair_mat.mat').replace("\\","/")

	bet_t1_2_flair = os.path.join(path_patient, 'bet_t1_2_flair.nii.gz').replace("\\","/")
	t1_2_std = os.path.join(path_patient, 't1_2_std.nii.gz').replace("\\","/")
	t1_2_std_mat = os.path.join(path_patient, 't1_2_std_mat.mat').replace("\\","/")
	transformation = os.path.join(path_patient, 'transformation.nii.gz').replace("\\","/")

	warped_t1 = os.path.join(path_patient, 'warped_t1.nii.gz').replace("\\","/")
	warped_fl = os.path.join(path_patient, 'warped_fl.nii.gz').replace("\\","/")
	warped_wmh = os.path.join(path_patient, 'warped_wmh.nii.gz').replace("\\","/")

	commands = list()

	# Skull removals
	# commands.append([f'{start}bet {mnt}{t1_path} {mnt}{bet_t1}"', 'T1 skull removal'])
	# commands.append([f'{start}bet {mnt}{fl_path} {mnt}{bet_fl}"', 'FL skull removal'])

	# Map T1 to FLAIR
	commands.append([f'{start}flirt -in {mnt}{t1_path} -ref {mnt}{fl_path} -out {mnt}{t1_2_flair} -omat {mnt}{t1_2_flair_mat} -bins 256 -cost mutualinfo -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 6  -interp trilinear"', 'Original T1 to FLAIR space (FLIRT)'])
	
	# Map t1_2_flair to MNI (4 steps)
	commands.append([f'{start}bet {mnt}{t1_2_flair} {mnt}{bet_t1_2_flair}"', '-1/4- Map t1_2_flair to MNI (BET)'])
	commands.append([f'{start}flirt -in {mnt}{bet_t1_2_flair} -ref {mnt}{mni_bet} -out {mnt}{t1_2_std} -omat {mnt}{t1_2_std_mat} -cost mutualinfo"', '-2/4- Map t1_2_flair to MNI (FLIRT)'])
	commands.append([f'{start}fnirt --in={mnt}{t1_2_flair} --aff={mnt}{t1_2_std_mat} --cout={mnt}{transformation} --ref={mnt}{mni_t1}"', '-3/4- Map t1_2_flair to MNI (FNIRT)'])
	commands.append([f'{start}applywarp --ref={mnt}{mni_t1} --in={mnt}{t1_2_flair} --warp={mnt}{transformation} --out={mnt}{warped_t1}"', '-4/4- Map t1_2_flair to MNI (APPLYWARP)'])
	
	# Map FL and label to MNI
	commands.append([f'{start}applywarp --ref={mnt}{mni_t1} --in={mnt}{fl_path} --warp={mnt}{transformation} --out={mnt}{warped_fl}"', 'Map FL to MNI (APPLYWARP)'])
	commands.append([f'{start}applywarp --ref={mnt}{mni_t1} --in={mnt}{lbl_path} --warp={mnt}{transformation} --out={mnt}{warped_wmh}"', 'Map label to MNI (APPLYWARP)'])

	for command,description in commands:
		print(f'   --- {description} ---')
		os.system(command)

	
def normalize_image(image):
	min_val, max_val = np.min(image), np.max(image)
	range = max_val - min_val
	return (image - min_val) / range