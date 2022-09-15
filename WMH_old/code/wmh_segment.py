import os
import numpy as np
import sys
import math
import time
import datetime
import nibabel as nib
from skimage.transform import resize
import theano
import wmh_network
import wmh_utility



def segment_it(X1, X2, X3, location_features, pos, start, f, shape, result, max_fold_size):
	[w, h, d] = shape
	patch_num = X1.shape[0]
	nr_folds = int(math.ceil(patch_num/float(max_fold_size)))
	#print "---------------------------",X1.shape, nr_folds
	for i in range(nr_folds):
		X1_sub = X1[i * max_fold_size: min((i+1) * max_fold_size, patch_num)]
		X2_sub = X2[i * max_fold_size: min((i+1) * max_fold_size, patch_num)]
		X3_sub = X3[i * max_fold_size: min((i+1) * max_fold_size, patch_num)]
		loc_feat_sub = location_features[i * max_fold_size: min((i+1) * max_fold_size, patch_num)]
		position_sub = pos[i * max_fold_size: min((i+1) * max_fold_size, patch_num)]
		pred_sub = f(X1_sub, X2_sub, X3_sub, loc_feat_sub)
		for j in range(X1_sub.shape[0]):    
			result[position_sub[j,0],position_sub[j,1],position_sub[j,2]+start] = pred_sub[j,1]

def get_coord(ref_size, case_scale, x, y, padding):
	return [int(x*float(ref_size)/case_scale)+padding, int(y*float(ref_size)/case_scale)+padding]

def scaleImages(images, ps):
	padding = ps[0]
	res = []	
	[w, h, d] = images[0].shape
	for ps_index in range(1, len(ps)):
		for image_index in range(len(images)):
			scaled = resize( images[image_index]/1000.0, ( int(w*float(ps[0])/ps[ps_index]), int(h*float(ps[0])/ps[ps_index]), d ) )
			res.append(np.lib.pad(scaled, ((padding, padding),(padding, padding),(0,0)), 'constant', constant_values=0)*1000.0)
			#pad_image(scaled, padding, padding, padding, padding)
	return res
			

def create_patches_to_be_classified(images, mask, selection_mask, loc_feature_Images, patch_sizes,  start, end):
	scaled_images = scaleImages(images, patch_sizes)
	patchSize = patch_sizes[0]
	half_patch = patchSize/2
	shape = images[0].shape
	nr_modalities = len(images)
	indexes = np.transpose(np.nonzero(mask[:,:,start:end]*selection_mask[:,:,start:end]))
	#indexes[:,2] += start
	l = indexes.shape[0]
	X1 = np.zeros((l, len(images), patchSize, patchSize), dtype='float32')
	X2 = np.zeros((l, len(images), patchSize, patchSize), dtype='float32')
	X3 = np.zeros((l, len(images), patchSize, patchSize), dtype='float32')
	loc_features = np.zeros((l, len(loc_feature_Images)), dtype='float32')
	
	for i in range(l):
		for counter in range(nr_modalities):
			X1[i,counter,:,:] = (images[counter])[indexes[i,0]-half_patch:indexes[i,0]+half_patch,indexes[i,1]-half_patch:indexes[i,1]+half_patch, indexes[i,2]]
			[x, y] = get_coord(patch_sizes[0], patch_sizes[1], indexes[i,0], indexes[i,1], patchSize)
			X2[i,counter,:,:] = (scaled_images[0+counter])[x-half_patch:x+half_patch, y-half_patch:y+half_patch, indexes[i,2]]
			[x, y] = get_coord(patch_sizes[0], patch_sizes[2], indexes[i,0], indexes[i,1], patchSize)
			X3[i,counter,:,:] = (scaled_images[nr_modalities+counter])[x-half_patch:x+half_patch,y-half_patch:y+half_patch, indexes[i,2]]

		for counter in range(len(loc_feature_Images)):
			loc_features[i, counter] = (loc_feature_Images[counter])[indexes[i,0],indexes[i,1], indexes[i,2]]

	return [X1, X2, X3, loc_features, indexes]


def run_batch(data_path, cases , model_path, lpd_path, out_file_name):
	f, lp, arch = wmh_network.load_network(lpd_path, model_path)
	locScales = [[0.0, 70.0],[0.0, 84.0],[0.0, 83.0],[0.0, 148.0],[0.0, 186.0],[0.0, 153.0],[0.0, 1.0]]
	max_fold_size = 1000
	nr_slices_each_time = 2
	for c in cases:
		oldNow = datetime.datetime.now()
		output_image = data_path+c+"/"+out_file_name	
		imgs =[]
		loc_feat_images=[]
		mask = nib.load(data_path+c+"/"+lp["mask"])
		mask_image = mask.get_data()
		for ff in arch["channels"]:
			imgs.append(nib.load(data_path+c+"/"+ff).get_data())
		for counter in range(len(lp["locationFeatures"])):
			loc_feat_images.append( (nib.load(data_path+c+"/"+lp["locationFeatures"][counter]).get_data()-locScales[counter][0])/(locScales[counter][1]-locScales[counter][0]) )
		sh = mask_image.shape
		result = np.zeros(sh, dtype='float')
		selection_mask = np.zeros(sh)
		margin = arch["patchSizes"][-1]//2
		selection_mask[margin:sh[0]-margin-1,margin:sh[1]-margin-1, :] = 1
		#print "getting the boundaries.."


		start = 0
		while start<sh[2]:
			end = min(start+nr_slices_each_time, sh[2])
			print("for slice",start, "to", end)
			sub_images=[]
			for img in imgs:
				sub_images.append(img[:,:, start:end])
			[X1, X2, X3, loc_features, pos] = create_patches_to_be_classified(sub_images, mask_image, selection_mask, loc_feat_images, arch["patchSizes"], start, end)
			if (X1.shape)[0]==0:
				start += nr_slices_each_time
				continue
			segment_it(X1, X2, X3, loc_features, pos, start, f, sh, result, max_fold_size)
			start += nr_slices_each_time
		nib.save(nib.Nifti1Image(result, mask.affine), output_image)	
		newNow = datetime.datetime.now()
		print(newNow-oldNow)

