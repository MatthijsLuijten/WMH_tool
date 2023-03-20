import os
import parameters
import nibabel as nib
import numpy as np
import time
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from plot import *
from utils import normalize_image, original_or_flirt


def preprocess_in_vivo_data(train_cases, test_cases):
    # Make train and test datasets
	train_img, train_lbl, test_img, test_lbl = [], [], [], []

	# Preprocess training cases
	for case in tqdm(train_cases):
		t1, fl, lbl, _ = preprocess(case)
		for slice in range(len(t1[0][0])):
			if slice >= 39 and slice <= 149:
				train_img.append(np.dstack((t1[:,:,slice], fl[:,:,slice])))
				train_lbl.append(np.reshape(lbl[:,:,slice], (182,218,1)))
				
	# Preprocess test cases
	for case in tqdm(test_cases):
		t1, fl, lbl, _ = preprocess(case)
		for slice in range(len(t1[0][0])):
			if slice > 39 and slice < 149:
				test_img.append(np.dstack((t1[:,:,slice], fl[:,:,slice])))
				test_lbl.append(np.reshape(lbl[:,:,slice], (182,218,1)))

	# Convert lists to numpy arrays
	train_img, train_lbl, test_img, test_lbl = map(np.array, (train_img, train_lbl, test_img, test_lbl))


	return train_img, train_lbl, test_img, test_lbl


def preprocess(c):
	# Determine which path to use based on the case name
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
	fl_img = fl_orig.get_fdata()
	lbl_img = lbl_orig.get_fdata()
	
	# Normalize to 0-1
	t1_norm = normalize_image(t1_img)
	fl_norm = normalize_image(fl_img)
	lbl_norm = normalize_image(lbl_img)

	# Threshold label to 0 or 1
	lbl_norm = np.where(lbl_norm > 0.1, 1., 0.)

	return t1_norm, fl_norm, lbl_norm, fl_orig.affine


def do_fsl(path_patient, t1_path, fl_path, lbl_path):
	startTime = time.time()
	wsl = 'wsl ~ -e sh -c'
	fsl = '/usr/local/fsl/bin/'
	# mnt = '/mnt/c/Users/Matthijs/Documents/RU/MSc_Thesis/WMH_tool/'
	mnt = '/mnt/c/Research/Matthijs/WMH_tool/'
	start = f'{wsl} "export FSLDIR=/usr/local/fsl; . ${{FSLDIR}}/etc/fslconf/fsl.sh; {fsl}'
	mni_bet = 'WMH_new/material/MNI152_T1_1mm_brain.nii.gz'
	
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

	
def preprocess_pm_data(train_cases, test_cases):
    # Make train and test datasets
	train_img, train_lbl_wmh, train_lbl_nawm, train_lbl_gm, test_img, test_lbl_wmh, test_lbl_nawm, test_lbl_gm = [], [], [], [], [], [], [], []

	# Preprocess training cases
	for case in tqdm(train_cases):
		t1, fl, lbl_wmh, lbl_nawm, lbl_gm = preprocess_pm(case)
		train_img.append(np.dstack((t1, fl)))
		train_lbl_wmh.append(np.reshape(lbl_wmh, (200,200,1)))
		train_lbl_nawm.append(np.reshape(lbl_nawm, (200,200,1)))
		train_lbl_gm.append(np.reshape(lbl_gm, (200,200,1)))
				
	# Preprocess test cases
	for case in tqdm(test_cases):
		t1, fl, lbl_wmh, lbl_nawm, lbl_gm = preprocess_pm(case)
		test_img.append(np.dstack((t1, fl)))
		test_lbl_wmh.append(np.reshape(lbl_wmh, (200,200,1)))
		test_lbl_nawm.append(np.reshape(lbl_nawm, (200,200,1)))
		test_lbl_gm.append(np.reshape(lbl_gm, (200,200,1)))

	# Convert lists to numpy arrays
	train_img = np.array(train_img)
	train_lbl_wmh = np.array(train_lbl_wmh)
	train_lbl_nawm = np.array(train_lbl_nawm)
	train_lbl_gm = np.array(train_lbl_gm)
	test_img = np.array(test_img)
	test_lbl_wmh = np.array(test_lbl_wmh)
	test_lbl_nawm = np.array(test_lbl_nawm)
	test_lbl_gm = np.array(test_lbl_gm)

	return train_img, train_lbl_wmh, train_lbl_nawm, train_lbl_gm, test_img, test_lbl_wmh, test_lbl_nawm, test_lbl_gm


def preprocess_pm_data_lfb(cases):
    # Make train and test datasets
	input_img, train_lbl_wmh, train_lbl_nawm, train_lbl_gm, lfb_img = [], [], [], [], []

	# Preprocess training cases
	for case in tqdm(cases):
		t1, fl, lfb, lbl_wmh, lbl_nawm, lbl_gm = preprocess_pm_lfb(case)
		input_img.append(np.dstack((t1, fl)))
		lfb_img.append(lfb)
		train_lbl_wmh.append(np.reshape(lbl_wmh, (200,200,1)))
		train_lbl_nawm.append(np.reshape(lbl_nawm, (200,200,1)))
		train_lbl_gm.append(np.reshape(lbl_gm, (200,200,1)))

	# Convert lists to numpy arrays
	input_img = np.array(input_img)
	train_lbl_wmh = np.array(train_lbl_wmh)
	train_lbl_nawm = np.array(train_lbl_nawm)
	train_lbl_gm = np.array(train_lbl_gm)
	lfb_img = np.array(lfb_img)

	return input_img, lfb_img


def preprocess_pm(c):
	# print('--> Case', c)
	path_patient = os.path.join(parameters.path_pm_wmh_data, c).replace("\\","/")

	# Path --> nifti image
	t1 = nib.load(os.path.join(path_patient, 't1_2_flair.nii.gz').replace("\\","/"))
	fl = nib.load(os.path.join(path_patient, 'fl.nii.gz').replace("\\","/"))
	
	# Nifti --> np array
	t1_img = t1.get_fdata()
	fl_img = fl.get_fdata()
		
	wmh_lbl = Image.open(os.path.join(path_patient, 'Data_Complete_'+ c +'_WMH.tiff').replace("\\","/")).convert('L')
	nawm_lbl = Image.open(os.path.join(path_patient, 'Data_Complete_'+ c +'_NAWM.tiff').replace("\\","/")).convert('L')
	gm_lbl = Image.open(os.path.join(path_patient, 'Data_Complete_'+ c +'_GM.tiff').replace("\\","/")).convert('L')
	
	# Rescale image
	t1_rescaled = np.array(Image.fromarray(t1_img).resize((200,200), resample=Image.Resampling.BICUBIC))
	fl_rescaled = np.array(Image.fromarray(fl_img).resize((200,200), resample=Image.Resampling.BICUBIC))
	wmh_lbl_rescaled = np.array(wmh_lbl.resize((200,200), resample=Image.Resampling.BICUBIC))
	nawm_lbl_rescaled = np.array(nawm_lbl.resize((200,200), resample=Image.Resampling.BICUBIC))
	gm_lbl_rescaled = np.array(gm_lbl.resize((200,200), resample=Image.Resampling.BICUBIC))

	# Normalize to 0-1
	t1_norm = normalize_image(t1_rescaled)
	fl_norm = normalize_image(fl_rescaled)
	wmh_lbl_norm = normalize_image(wmh_lbl_rescaled)
	nawm_lbl_norm = normalize_image(nawm_lbl_rescaled)
	gm_lbl_norm = normalize_image(gm_lbl_rescaled)
	
	# Threshold label to 0 or 1
	wmh_lbl_norm = np.where(np.array(wmh_lbl_norm) > 0.1, 1., 0.)
	nawm_lbl_norm = np.where(np.array(nawm_lbl_norm) > 0.1, 1., 0.)
	gm_lbl_norm = np.where(np.array(gm_lbl_norm) > 0.1, 1., 0.)
	return t1_norm, fl_norm, wmh_lbl_norm, nawm_lbl_norm, gm_lbl_norm


def preprocess_pm_lfb(c):
	# print('--> Case', c)
	path_patient = os.path.join(parameters.path_pm_wmh_data, c).replace("\\","/")

	# Path --> nifti image
	t1 = nib.load(os.path.join(path_patient, 't1_2_flair.nii.gz').replace("\\","/"))
	fl = nib.load(os.path.join(path_patient, 'fl.nii.gz').replace("\\","/"))
	lfb = nib.load(os.path.join(path_patient, 'lfb_2_flair.nii.gz').replace("\\","/"))
	
	# Nifti --> np array
	t1_img = t1.get_fdata()
	fl_img = fl.get_fdata()
	orig_lfb = Image.open(os.path.join(path_patient, c +'_LFB_comparabletoMRIresolution_cleaned.tif').replace("\\","/")).convert('L')
		
	wmh_lbl = Image.open(os.path.join(path_patient, 'Data_Complete_'+ c +'_WMH.tiff').replace("\\","/")).convert('L')
	nawm_lbl = Image.open(os.path.join(path_patient, 'Data_Complete_'+ c +'_NAWM.tiff').replace("\\","/")).convert('L')
	gm_lbl = Image.open(os.path.join(path_patient, 'Data_Complete_'+ c +'_GM.tiff').replace("\\","/")).convert('L')
	
	# Rescale image
	t1_rescaled = np.array(Image.fromarray(t1_img).resize((200,200), resample=Image.Resampling.BICUBIC))
	fl_rescaled = np.array(Image.fromarray(fl_img).resize((200,200), resample=Image.Resampling.BICUBIC))
	orig_lfb = np.array(orig_lfb.resize((200,200), resample=Image.Resampling.BICUBIC))
	wmh_lbl_rescaled = np.array(wmh_lbl.resize((200,200), resample=Image.Resampling.BICUBIC))
	nawm_lbl_rescaled = np.array(nawm_lbl.resize((200,200), resample=Image.Resampling.BICUBIC))
	gm_lbl_rescaled = np.array(gm_lbl.resize((200,200), resample=Image.Resampling.BICUBIC))

	# Normalize to 0-1
	t1_norm = normalize_image(t1_rescaled)
	fl_norm = normalize_image(fl_rescaled)
	orig_lfb = normalize_image(orig_lfb)
	orig_lfb = np.where(orig_lfb == 1, 0, orig_lfb)
	wmh_lbl_norm = normalize_image(wmh_lbl_rescaled)
	nawm_lbl_norm = normalize_image(nawm_lbl_rescaled)
	gm_lbl_norm = normalize_image(gm_lbl_rescaled)
	
	# Threshold label to 0 or 1
	wmh_lbl_norm = np.where(np.array(wmh_lbl_norm) > 0.1, 1., 0.)
	nawm_lbl_norm = np.where(np.array(nawm_lbl_norm) > 0.1, 1., 0.)
	gm_lbl_norm = np.where(np.array(gm_lbl_norm) > 0.1, 1., 0.)

	return t1_norm, fl_norm, orig_lfb, wmh_lbl_norm, nawm_lbl_norm, gm_lbl_norm


def preprocess_pm_inference(cases):
    # Make train and test datasets
	input_img = []

	# Preprocess training cases
	for case in tqdm(cases):
		path_patient = os.path.join(parameters.path_pm_wmh_data, case).replace("\\","/")
		print(path_patient)
		t1 = nib.load(os.path.join(path_patient, 't1_2_flair.nii.gz').replace("\\","/"))
		fl = nib.load(os.path.join(path_patient, 'fl.nii.gz').replace("\\","/"))

		t1_img = t1.get_fdata()
		fl_img = fl.get_fdata()

		t1_rescaled = np.array(Image.fromarray(t1_img).resize((200,200), resample=Image.Resampling.BICUBIC))
		fl_rescaled = np.array(Image.fromarray(fl_img).resize((200,200), resample=Image.Resampling.BICUBIC))

		t1_norm = normalize_image(t1_rescaled)
		fl_norm = normalize_image(fl_rescaled)
		
		input_img.append(np.dstack((t1_norm, fl_norm)))
		
	# Convert lists to numpy arrays
	input_img = np.array(input_img)

	return input_img


def preprocess_pm_wmh(c, data_path):
	patient_path = os.path.join(data_path, c).replace("\\","/")
	
	for filename in os.listdir(patient_path):
		f = os.path.join(patient_path, filename).replace("\\","/")
		
		if 'T1' in filename and 'RAW' not in filename:
			img = Image.open(os.path.join(patient_path, f).replace("\\","/")).convert('L')
			img = img.resize((200,200), resample=Image.Resampling.NEAREST)
			img = np.asarray(img)
			nii_img = nib.Nifti1Image(img, affine=np.eye(4))
			nib.save(nii_img, os.path.join(patient_path, 't1.nii.gz').replace("\\","/"))

		elif 'FL' in filename and 'RAW' not in filename:
			img = Image.open(os.path.join(patient_path, f).replace("\\","/")).convert('L')
			img = img.resize((200,200), resample=Image.Resampling.NEAREST)
			img = np.asarray(img)
			nii_img = nib.Nifti1Image(img, affine=np.eye(4))
			nib.save(nii_img, os.path.join(patient_path, 'fl.nii.gz').replace("\\","/"))

		# elif 'LFB_comparabletoMRIresolution_8bit_cleaned' in filename:
		# 	img = Image.open(os.path.join(patient_path, f).replace("\\","/")).convert('L')
		# 	# img = img.resize((200,200), resample=Image.Resampling.BICUBIC)
		# 	img = np.asarray(img)
		# 	img = np.where(img ==255, 0, img)
		# 	nii_img = nib.Nifti1Image(img, affine=np.eye(4))
		# 	nib.save(nii_img, os.path.join(patient_path, 'lfb.nii.gz').replace("\\","/"))

	do_pm_wmh_fsl(c)

def do_pm_wmh_fsl(path_patient):
	wsl = 'wsl ~ -e sh -c'
	fsl = '/usr/local/fsl/bin/'
	# mnt = '/mnt/e/Matthijs/postmortem_WMH/'
	mnt = '/mnt/d/Gemma/C338C_MRI/'
	start = f'{wsl} "export FSLDIR=/usr/local/fsl; . ${{FSLDIR}}/etc/fslconf/fsl.sh; {fsl}'

	t1 = os.path.join(path_patient, 't1.nii.gz').replace("\\","/")
	fl = os.path.join(path_patient, 'fl.nii.gz').replace("\\","/")
	lfb = os.path.join(path_patient, 'lfb.nii.gz').replace("\\","/")

	t1_2_flair = os.path.join(path_patient, 't1_2_flair.nii.gz').replace("\\","/")
	t1_2_flair_mat = os.path.join(path_patient, 't1_2_flair_mat.mat').replace("\\","/")

	lfb_2_flair = os.path.join(path_patient, 'lfb_2_flair.nii.gz').replace("\\","/")
	lfb_2_flair_mat = os.path.join(path_patient, 'lfb_2_flair.mat').replace("\\","/")

	commands = list()

	commands.append([f'{start}flirt -in {mnt}{t1} -ref {mnt}{fl} -out {mnt}{t1_2_flair} -omat {mnt}{t1_2_flair_mat} -2D -schedule /usr/local/fsl/etc/flirtsch/sch2D_6dof', 'T1 to FLAIR space (FLIRT)'])
	# commands.append([f'{start}flirt -in {mnt}{lfb} -ref {mnt}{fl} -out {mnt}{lfb_2_flair} -omat {mnt}{lfb_2_flair_mat} -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -2D -dof 12  -interp trilinear', 'LFB to FLAIR space (FLIRT)'])

	for command,description in commands:
		print(description)
		os.system(command)