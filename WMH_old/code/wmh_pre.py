import os
import sys
import re
import multiprocessing
from  multiprocessing import Pool
import time
import os.path



def worker(atlas_path, patientPath, files, fsl_version, label_file):
	
	startTime = time.time()
	imagesPath = patientPath
	#original files
	T1OrigPath = os.path.join(imagesPath, files[1])
	FLOrigPath = os.path.join(imagesPath, files[0])
	labelOrigPath = os.path.join(imagesPath, label_file) 

	#swapped files (reoriented)
	T1T1Path = os.path.join(imagesPath,'_t1_t1.nii.gz')
	FLFLPath = os.path.join(imagesPath,'_fl_fl.nii.gz')
	labelLabelPath = os.path.join(imagesPath, '_label_label.nii.gz')

	#bet stuff (Skull removed)
	T1BetT1 = os.path.join(imagesPath,'_t1_brain_t1')
	T1BetT1Path = os.path.join(imagesPath,'_t1_brain_t1.nii.gz')
	T1BetMaskT1 = os.path.join(imagesPath,'_t1_brain_t1_mask.nii.gz')
	brainMaskT1 = os.path.join(imagesPath,'_brainMask_t1.nii.gz')

	#linear registration
	T1FLPath = os.path.join(imagesPath,'_t1_reg_fl.nii.gz')
	T1FLmatrixPath = os.path.join(imagesPath,'_t1_reg_fl.mat')
	brainMaskFL = os.path.join(imagesPath,'_brainMask_fl.nii.gz')

	#brain images
	T1BrainFLPath = os.path.join(imagesPath,'_t1_brain_fl.nii.gz')
	FLBrainFLPath = os.path.join(imagesPath,'_fl_brain_fl.nii.gz')

	#fast (I guess for bias correction)
	FastBaseFL = os.path.join(imagesPath,'_fastOutput_fl')
	FastBaseT1 = os.path.join(imagesPath,'_fastOutput_T1')

	T1Biascorrectbefore = os.path.join(imagesPath,'_fastOutput_fl_restore_2.nii.gz')
	FlBiascorrectbefore = os.path.join(imagesPath,'_fastOutput_fl_restore_1.nii.gz')
	T1Biasbefore = os.path.join(imagesPath,'_fastOutput_fl_bias_2.nii.gz')
	FlBiasbefore = os.path.join(imagesPath,'_fastOutput_fl_bias_1.nii.gz')

	T1BiasCorrected = os.path.join(imagesPath,'_t1_biasCorrected_fl.nii.gz')
	FlBiasCorrected = os.path.join(imagesPath,'_fl_biasCorrected_fl.nii.gz')
	T1BiasField = os.path.join(imagesPath,'_t1_biasField_fl.nii.gz')
	FlBiasField = os.path.join(imagesPath,'_fl_biasField_fl.nii.gz')

	commands = list()
	#reorient to standard axial view
	commands.append([fsl_version + 'fslreorient2std ' + T1OrigPath + ' ' + T1T1Path, 'Reorient T1...\n'])
	commands.append([fsl_version + 'fslreorient2std ' + FLOrigPath + ' ' + FLFLPath, 'Reorient FL...\n'])
	commands.append([fsl_version + 'fslreorient2std ' + labelOrigPath + ' ' + labelLabelPath, 'Reorient labels...\n'])

	#brain extraction
	commands.append([fsl_version + 'bet ' + T1T1Path + ' ' + T1BetT1 + ' -v -m -R', 'Brain Extraction...\n'])
	#rename it
	commands.append(['mv ' + T1BetMaskT1 + ' ' + brainMaskT1, 'Moving brain mask...\n'])

	# Registration linear
	commands.append([fsl_version + 'flirt -cost mutualinfo -searchcost mutualinfo -in ' + T1T1Path + ' -ref ' + FLFLPath + ' -out ' + T1FLPath + ' -omat ' + T1FLmatrixPath, 'Linear Registration T1->FL...\n'])
	#transform brain mask
	commands.append([fsl_version + 'flirt -cost mutualinfo -searchcost mutualinfo -in ' + brainMaskT1 + ' -ref ' + FLFLPath + ' -applyxfm -init ' + T1FLmatrixPath + ' -out ' + brainMaskFL , 'Transforming brain mask...\n'])

	#apply brain masks
	commands.append([fsl_version + 'fslmaths ' + T1FLPath + ' -mas ' + brainMaskFL + ' ' + T1BrainFLPath, 'Mask T1...\n'])
	commands.append([fsl_version + 'fslmaths ' + FLFLPath + ' -mas ' + brainMaskFL + ' ' + FLBrainFLPath, 'Mask FL...\n'])

	#segmentation and bias field correction in FL space
	commands.append([fsl_version + 'fast -v -S 2 -t 1 -n 3 --nopve -B -b -o ' + FastBaseFL + ' ' + T1BrainFLPath  + ' ' + FLBrainFLPath, 'Bias Field Correction...\n'])


	commands.append(['mv ' + T1Biascorrectbefore + ' ' + T1BiasCorrected, 'Rename T1 bias corrected...\n'])
	commands.append(['mv ' + T1Biasbefore + ' ' + T1BiasField, 'Rename T1 bias field...\n'])
	commands.append(['mv ' + FlBiascorrectbefore + ' ' + FlBiasCorrected, 'Rename FL bias corrected...\n'])
	commands.append(['mv ' + FlBiasbefore + ' ' + FlBiasField, 'Rename FL bias field...\n'])

	#segmentation and bias field correction in T1 space
	commands.append([fsl_version + 'fast -v -S 1 -t 1 -n 3 --nopve -B -b -o ' + FastBaseT1 + ' ' + T1BetT1Path, 'T1 Bias Field Correction...\n'])

	for command,description in commands:
		print("\033[91m",description,"\033[0m")
		os.system(command)

	#--------------------------------------------------------------------------Run (reorientaion+skul removal+registration+bias_feild_correction)
	commands = list()	
	#== Non-Linear registration =====================================
	T1AtlasPath =atlas_path+'MNI152_T1_2mm.nii.gz'
	T1BrainAtlasPath =atlas_path+'MNI152_T1_2mm_brain.nii.gz'
	refMask = atlas_path+'MNI152_T1_2mm_brain_mask_dil.nii.gz'
	FnirtCFGPath = atlas_path+'T1_2_MNI152_2mm.cnf'
	

	t1Atlas = os.path.join(imagesPath,'_tmp_Atlas_T1_2mm.nii.gz')
	t1BrainAtlas = os.path.join(imagesPath,'_tmp_Atlas_T1_2mm_brain.nii.gz')

	t12mm = os.path.join(imagesPath,'_tmp_t1_2mm.nii.gz')
	t1Bet2mm = os.path.join(imagesPath,'_tmp_t1_2mm_Brain.nii.gz')

	t1RegAtlas = os.path.join(imagesPath,'_t1_reg_atlas.nii.gz')
	t1RegAtlasMat = os.path.join(imagesPath,'_t1_reg_atlas.mat')
	atlasRegT1Mat = os.path.join(imagesPath,'_atlas_reg_t1.mat')

	t1RegAtlasWarp = os.path.join(imagesPath,'_t1_reg_atlas_warp.nii.gz')
	atlasRegT1Warp = os.path.join(imagesPath,'_atlas_reg_t1_warp.nii.gz')

	atlasRegT1 = os.path.join(imagesPath,'_atlas_reg_t1.nii.gz')
	atlasRegFL = os.path.join(imagesPath,'_reg_fl.nii.gz')
	AtlasFL = os.path.join(imagesPath,'_MNI_FL.nii.gz')


	#commands.append(['cp ' + T1AtlasPath + ' ' + t1Atlas, 'Copy T1 Atlas...\n'])
	#commands.append(['cp ' + T1BrainAtlasPath + ' ' + t1BrainAtlas, 'Copy T1 Brain Atlas...\n'])

	#flirt t1 to atlas
	commands.append([fsl_version + 'flirt -v -cost mutualinfo -searchcost mutualinfo -in ' + T1BetT1Path + ' -ref ' + T1BrainAtlasPath + ' -out ' + t1RegAtlas + ' -omat ' + t1RegAtlasMat, 'Linear Registration T1->Atlas...\n'])

	#invert transformation
	commands.append([fsl_version + 'convert_xfm -omat ' + atlasRegT1Mat + ' -inverse ' + t1RegAtlasMat, 'Invert Transformation Atlas->T1...\n'])

	# fnirt t1 to t1Atlas
	commands.append([fsl_version + 'fnirt -v --ref=' + T1BrainAtlasPath + ' --in=' + t1RegAtlas + ' --config=' + FnirtCFGPath + ' --cout=' + t1RegAtlasWarp + ' --refmask='+refMask, 'Non-Linear Registration T1->Atlas...\n'])

	# invert warp (to get atlas to t1)
	commands.append([fsl_version + 'invwarp -w ' + t1RegAtlasWarp + ' -o ' + atlasRegT1Warp + ' -r ' + t1RegAtlas, 'Invert Warp Atlas->T1...\n'])

	#apply warp to atlas (to get warped t1 atlas)
	commands.append([fsl_version + 'applywarp -i ' + T1BrainAtlasPath + ' -o ' + atlasRegT1 + ' -r ' + t1RegAtlas + ' -w ' + atlasRegT1Warp, 'Apply warp to Atlas...\n'])
	#apply 1st transformation to atlas
	commands.append([fsl_version + 'flirt -cost mutualinfo -searchcost mutualinfo -in ' + atlasRegT1 + ' -ref ' + T1BetT1Path + ' -applyxfm -init ' + atlasRegT1Mat + ' -out ' + atlasRegFL, 'Apply transformation 1 to Atlas...\n'])
	#apply 2nd transformation to atlas
	commands.append([fsl_version + 'flirt -cost mutualinfo -searchcost mutualinfo -in ' + atlasRegFL + ' -ref ' + FLFLPath + ' -applyxfm -init ' + T1FLmatrixPath + ' -out ' + AtlasFL, 'Apply transformation 2 to Atlas...\n'])

	#== END Non-Linear registration =====================================

	for command,description in commands:
		print("\033[91m",description,"\033[0m")
		os.system(command)
	commands = list()

	#++++++++++++++++++++++++++++++

	MNIMasksPath = atlas_path + 'MasksUnzipped/'
	XPosImage = os.path.join(MNIMasksPath, 'MNIXPositions_corrected.nii')
	XPosHalfReg = os.path.join(imagesPath,'XPos_halfreg_t1.nii.gz')
	XPosFluidFL = os.path.join(imagesPath,'XPos_fl.nii.gz')
	XPosFluidT1 = os.path.join(imagesPath,'XPos_t1.nii.gz')

	YPosImage = os.path.join(MNIMasksPath, 'MNIYPositions_corrected.nii')
	YPosHalfReg = os.path.join(imagesPath,'YPos_halfreg_t1.nii.gz')
	YPosFluidFL = os.path.join(imagesPath,'YPos_fl.nii.gz')
	YPosFluidT1 = os.path.join(imagesPath,'YPos_t1.nii.gz')

	ZPosImage = os.path.join(MNIMasksPath, 'MNIZPositions_corrected.nii')
	ZPosHalfReg = os.path.join(imagesPath,'ZPos_halfreg_t1.nii.gz')
	ZPosFluidFL = os.path.join(imagesPath,'ZPos_fl.nii.gz')
	ZPosFluidT1 = os.path.join(imagesPath,'ZPos_t1.nii.gz')


	atlasImage = os.path.join(MNIMasksPath, 'wmh_atlas.nii')
	atlasHalfReg = os.path.join(imagesPath,'wmh_atlas_halfreg_t1.nii.gz')
	atlasFluidFL = os.path.join(imagesPath,'wmh_atlas_fl.nii.gz')
	atlasFluidT1 = os.path.join(imagesPath,'wmh_atlas_t1.nii.gz')


	#apply warp to xpos
	commands.append([fsl_version + 'applywarp -i ' + XPosImage + ' -o ' + XPosHalfReg + ' -r ' + t1RegAtlas + ' -w ' + atlasRegT1Warp, 'Apply warp to xpos...\n'])
	#apply 1st transformation to xpos
	commands.append([fsl_version + 'flirt -cost mutualinfo -searchcost mutualinfo -in ' + XPosHalfReg + ' -ref ' + T1BetT1Path + ' -applyxfm -init ' + atlasRegT1Mat + ' -out ' + XPosFluidT1, 'Apply transformation 1 to xpos...\n'])
	#apply 2nd transformation to xpos
	commands.append([fsl_version + 'flirt -cost mutualinfo -searchcost mutualinfo -in ' + XPosFluidT1 + ' -ref ' + FLFLPath + ' -applyxfm -init ' + T1FLmatrixPath + ' -out ' + XPosFluidFL, 'Apply transformation 2 to xpos...\n'])

	#apply warp to ypos
	commands.append([fsl_version + 'applywarp -i ' + YPosImage + ' -o ' + YPosHalfReg + ' -r ' + t1RegAtlas + ' -w ' + atlasRegT1Warp, 'Apply warp to ypos...\n'])
	#apply 1st transformation to ypos
	commands.append([fsl_version + 'flirt -cost mutualinfo -searchcost mutualinfo -in ' + YPosHalfReg + ' -ref ' + T1BetT1Path + ' -applyxfm -init ' + atlasRegT1Mat + ' -out ' + YPosFluidT1, 'Apply transformation 1 to ypos...\n'])
	#apply 2nd transformation to ypos
	commands.append([fsl_version + 'flirt -cost mutualinfo -searchcost mutualinfo -in ' + YPosFluidT1 + ' -ref ' + FLFLPath + ' -applyxfm -init ' + T1FLmatrixPath + ' -out ' + YPosFluidFL, 'Apply transformation 2 to ypos...\n'])

	#apply warp to zpos
	commands.append([fsl_version + 'applywarp -i ' + ZPosImage + ' -o ' + ZPosHalfReg + ' -r ' + t1RegAtlas + ' -w ' + atlasRegT1Warp, 'Apply warp to zpos...\n'])
	#apply 1st transformation to zpos
	commands.append([fsl_version + 'flirt -cost mutualinfo -searchcost mutualinfo -in ' + ZPosHalfReg + ' -ref ' + T1BetT1Path + ' -applyxfm -init ' + atlasRegT1Mat + ' -out ' + ZPosFluidT1, 'Apply transformation 1 to zpos...\n'])
	#apply 2nd transformation to zpos
	commands.append([fsl_version + 'flirt -cost mutualinfo -searchcost mutualinfo -in ' + ZPosFluidT1 + ' -ref ' + FLFLPath + ' -applyxfm -init ' + T1FLmatrixPath + ' -out ' + ZPosFluidFL, 'Apply transformation 2 to zpos...\n'])

	#apply warp to wmh atlas
	commands.append([fsl_version + 'applywarp -i ' + atlasImage + ' -o ' + atlasHalfReg + ' -r ' + t1RegAtlas + ' -w ' + atlasRegT1Warp, 'Apply warp to wmh atlas...\n'])
	#apply 1st transformation to wmh atlas
	commands.append([fsl_version + 'flirt -cost mutualinfo -searchcost mutualinfo -in ' + atlasHalfReg + ' -ref ' + T1BetT1Path + ' -applyxfm -init ' + atlasRegT1Mat + ' -out ' + atlasFluidT1, 'Apply transformation 1 to wmh atlas...\n'])
	#apply 2nd transformation to wmh atlas
	commands.append([fsl_version + 'flirt -cost mutualinfo -searchcost mutualinfo -in ' + atlasFluidT1 + ' -ref ' + FLFLPath + ' -applyxfm -init ' + T1FLmatrixPath + ' -out ' + atlasFluidFL, 'Apply transformation 2 to wmh atlas...\n'])


	for command,description in commands:
		print("\033[91m",description,"\033[0m")
		os.system(command)
	totalTime = time.time()-startTime
	print("+++++++++++++took", totalTime/60.0, "mintues")
	
#'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''MAIN''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def run(data_path, atlas_path, c, files, label_file):
	fsl_version = 'fsl5.0-'
	patientPath = os.path.join(data_path, c)
	print('Starting patient', c, " >>")
	worker(atlas_path, patientPath, files, fsl_version, label_file)
	



