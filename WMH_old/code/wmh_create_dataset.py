import numpy as np
import random
from random import shuffle
import nibabel as nib
import sys
from skimage import transform as trans
from skimage.transform import resize
import h5py
import wmh_network
import wmh_utility
	
	
def create_initial_candidates(data_path, cases_list, neg_prob, pos_prob, files, margin, lp):
	print("Pass 1: Collecting data...")
	patches = list()
	nr_patients = len(cases_list)
	sh = nib.load(data_path+cases_list[0]+"/"+lp["mask"]).get_data().shape
	print(">>>>>>>>>>>>",sh)
	selection_mask = np.zeros(sh)
	selection_mask[margin:sh[0]-margin-1,margin:sh[1]-margin-1, :] = 1
	for i in range(nr_patients):
		patient_path = data_path+cases_list[i]+"/"
		mask_image = nib.load(patient_path+lp["mask"]).get_data()*selection_mask
		sh = mask_image.shape
		posImage = nib.load(patient_path+lp["label"]).get_data()*selection_mask
		mask_image = mask_image*(1-posImage)
		pos_indexes = np.transpose(np.nonzero(posImage))
		np.random.shuffle(pos_indexes)
		pos_indexes = pos_indexes[:int(pos_prob*pos_indexes.shape[0])]
		neg_indexes = np.transpose(np.nonzero(mask_image))
		np.random.shuffle(neg_indexes)
		neg_indexes = neg_indexes[:int(neg_prob*neg_indexes.shape[0])]
		pos_labels = np.ones((pos_indexes.shape[0], 1), dtype='int32')
		pos_indexes = np.concatenate((pos_indexes, pos_labels), axis=1)
		neg_labels = np.zeros((neg_indexes.shape[0], 1), dtype='int32')
		neg_indexes = np.concatenate((neg_indexes, neg_labels), axis=1)
		cur_cands = np.concatenate((pos_indexes, neg_indexes), axis=0)
		case_indexes = np.ones((cur_cands.shape[0], 1), dtype='int32')*i
		cur_cands = np.concatenate((cur_cands, case_indexes), axis=1)
		cur_cands = cur_cands.tolist()
		np.random.shuffle(cur_cands)
		pos = pos_indexes.shape[0]
		neg = neg_indexes.shape[0]
		print("pass 1 progress: "+ str(int((float(i+1)/nr_patients)*100.0))+"%")
		patches.extend(cur_cands)
		print("for patient " + str(i)+" pos:"+str(pos)+", neg:"+str(neg))
	return patches
	
	

def balance_dataset(patches):
	print("pass2: balancing the dataset")
	pos_aug = 1
	total_pos = total_neg = 0
	n = len(patches)
	
	for i in range(n):
		if patches[i][3]==0:
			total_neg += 1
		else:
			total_pos += 1
	print("initially +:"+str(total_pos)+" -:"+str(total_neg))
	ratio=float(total_pos)/float(total_neg)
	if total_pos>total_neg:
		ratio = 1.0/ratio
	temp = list()
	print("ratio is:"+str(ratio))
	for i in range(n):
		if patches[i][3]==0:
			if  total_pos<total_neg:
				r = random.random()
				if r<ratio:
					temp.append(patches[i])
			else:
				temp.append(patches[i])
		else:
			if total_pos>total_neg:
				r = random.random()
				if r<ratio:
					temp.append(patches[i])
			else:
				temp.append(patches[i])
	neg = pos = 0
	patches = temp
	n = len(patches)
	for i in range(n):
		if patches[i][3]==0:
			neg += 1
		else:
			pos += 1
	print("dataset after equalization, +:"+str(pos)+" -:"+str(neg))
	return [patches, pos, neg]


def create_and_save_datset(patches, data_path, patients_list, out_path, file_names, chunk_size, location_features, sizes, loc_scales):
	channels = len(file_names)
	loc_feature_file = open(out_path+"_loc.csv", "w")
	print("pass3: creating dataset...")
	cur_image = 0
	dataset_size = len(patches)
	rows = sizes[0]
	cols = sizes[0]
	path2 = data_path+patients_list[0]+"/"
	images = []
	location_images = []
	for ff in file_names:
		images.append((nib.load(path2+ff).get_data()))
	for ff in location_features:
		location_images.append((nib.load(path2+ff).get_data()))
	classes= [0, 0]
	step = dataset_size/100
	index = 0									#index counts the samples in each chunk

	f = h5py.File(out_path, 'w')
	total_features = f.create_dataset('patches', (dataset_size, channels*len(sizes), rows, cols),dtype="float32", compression='lzf', chunks=(chunk_size, channels*len(sizes), rows, cols ))
	total_labels = f.create_dataset('y', (dataset_size, 2), dtype="float32", compression='lzf', chunks=(chunk_size, 2))
	
	
	cur_chunk_features = np.empty((chunk_size, channels*len(sizes), rows, cols), dtype='float32')
	cur_chunk_labels = np.empty((chunk_size, 2), dtype='float32')
	counter = 0; 									#counting on [0...number of chunks]
	nr_location_features = len(location_images)
	for i in range (dataset_size):
		if i%step==0:
			print(str(int(float(i)/(float(dataset_size))*100))+"%")
		loc = patches[i]							
		if cur_image != loc[4]:						
			cur_image = loc[4]					
			path2 = data_path+patients_list[loc[4]]+"/"
			images = list()
			location_images = list()
			for ff in file_names:
				images.append((nib.load(path2+ff).get_data()))
			for ff in location_features:
				location_images.append((nib.load(path2+ff).get_data()))
		for sc in range(len(sizes)):
			for k in range(channels):		
				if sc==0:		#if this is the main dimension (32x32)
					scaledPatch = (images[k])[loc[0]-sizes[0]/2:loc[0]+sizes[0]/2, loc[1]-sizes[0]/2:loc[1]+sizes[0]/2, loc[2]]
				else:			#for other dimensions
					im=(images[k])[loc[0]-sizes[sc]/2:loc[0]+sizes[sc]/2, loc[1]-sizes[sc]/2:loc[1]+sizes[sc]/2, loc[2]]/100.0
					if im.shape[0]<32 or im.shape[1]<32:
						print("+++++++++++++++", k, loc )
					scaledPatch = resize(im, (sizes[0], sizes[0]))*100.0
				if scaledPatch.shape[0]!=32 or scaledPatch.shape[1]!=32:
					print("+++++++++++++++", k, loc )
				cur_chunk_features[index, sc*channels+k, :, :] = scaledPatch
		for k in range(nr_location_features-1):
			loc_feature_file.write(str((location_images[k][loc[0], loc[1], loc[2]]-loc_scales[k][0])/(loc_scales[k][1]-loc_scales[k][0]))+",")
		loc_feature_file.write(str(location_images[-1][loc[0], loc[1], loc[2]])+"\n")
		cur_chunk_labels[index,:] = [1-loc[3], loc[3]]
		index += 1
		classes[loc[3]] = classes[loc[3]] + 1
		if  i==dataset_size-1:		# if this is the last chunk
			print("saving for the last time! :D")
			total_features[counter*chunk_size:counter*chunk_size+index] = cur_chunk_features [0:index]
			total_labels[counter*chunk_size:counter*chunk_size+index] = cur_chunk_labels[0:index]
			index = 0;
			counter +=  1

		elif index >=chunk_size:		# if this is a regualr chunk
			print("saving chunk " +str(counter))
			total_features[counter*chunk_size:(counter+1)*chunk_size] = cur_chunk_features
			total_labels[counter*chunk_size:(counter+1)*chunk_size] = cur_chunk_labels
			index = 0;
			counter += 1
		
	print("pos "+str(classes[1]))
	print("neg "+str(classes[0]))
	loc_feature_file.close()
	f.close()






def run(nets_path, number, data_path):
	lpd_path = nets_path + number+"/"+number+".lpd"
	lp, arch, error = wmh_network.load_learning_process_description(lpd_path)


	
	loc_scales = [[0.0, 70.0],[0.0, 84.0],[0.0, 83.0],[0.0, 148.0],[0.0, 186.0],[0.0, 153.0],[0.0, 1.0]]
	channels = arch["channels"]
	nr_classes= arch["outClassNum"]
	neg_prob = 0.01
	pos_prob = 0.5
	chunk_size = lp["chunkSize"]
	training_cases = wmh_utility.load_cases(lp["trainingCases"])
	validation_cases = wmh_utility.load_cases(lp["validationCases"])


	print ("doing stuff for training set...")

	training_patches = create_initial_candidates(data_path, training_cases, neg_prob, pos_prob, channels, arch["patchSizes"][-1]//2+1, lp)
	[balanced_traing_patches, pos, neg] = balance_dataset(training_patches)
	create_and_save_datset(balanced_traing_patches, data_path, training_cases, lp["trainingSetPath"], channels, chunk_size, lp["locationFeatures"], arch["patchSizes"], loc_scales)

	print("doing stuff for validation set...")
	validation_patches = create_initial_candidates(data_path, validation_cases, neg_prob, pos_prob, channels, arch["patchSizes"][-1]//2+1, lp)
	[balanced_validation_patches, pos, neg] = balance_dataset(validation_patches)
	create_and_save_datset(balanced_validation_patches, data_path, validation_cases, lp["validationSetPath"], channels, chunk_size, lp["locationFeatures"], arch["patchSizes"], loc_scales)
	
	print(":))))) done creating the dataset...")
