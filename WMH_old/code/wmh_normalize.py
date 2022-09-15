import numpy as np
import nibabel as nib


def run(data_path, c):
	files = ['_fl_biasCorrected_fl.nii.gz', '_t1_biasCorrected_fl.nii.gz']
	for f in files:
		orig_im = nib.load(data_path+c+"/"+f)
		im = orig_im.get_data()
		mask = nib.load(data_path+c+"/_brainMask_fl.nii.gz").get_data()
		val = float(np.percentile(im[mask>0], 95))
		im = im/val
		nib.save(nib.Nifti1Image(im, orig_im.affine), data_path+c+"/nor"+f)


