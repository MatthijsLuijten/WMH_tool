import time
import sys
from turtle import Shape

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate

import math
import numpy as np
import nibabel as nib
from scipy import ndimage

#################### The U-Net Model ####################
#Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64, 0.1)
    s2, p2 = encoder_block(p1, 128, 0.1)
    s3, p3 = encoder_block(p2, 256, 0.1)
    s4, p4 = encoder_block(p3, 512, 0.1)

    b1 = conv_block(p4, 1024, 0.1) #Bridge

    d1 = decoder_block(b1, s4, 512, 0.1)
    d2 = decoder_block(d1, s3, 256, 0.1)
    d3 = decoder_block(d2, s2, 128, 0.1)
    d4 = decoder_block(d3, s1, 64, 0.1)

    outputs = Conv2D(1, 1, activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")
    return model

#################### The U-Net Model ####################\


#################### Building blocks ####################
def conv_block(input, num_filters, dropout_rate):
    x = Conv2D(num_filters, 3, activation='relu')(input)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)   #Not in the original network. 
    # x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)  #Not in the original network
    # x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling

def encoder_block(input, num_filters, dropout_rate):
    x = conv_block(input, num_filters, dropout_rate)
    p = MaxPool2D((2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters, dropout_rate):
    x = UpSampling2D((2, 2))(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, dropout_rate)
    return x

#################### Building blocks ####################

def loadValidFunction(X, Y, network, lp):
	print("		creating network output...")
	out = lasagne.layers.get_output(network)
	if lp["lastLayer"]=="sigmoid":
		prediction = out
	else:
		e_x = np.exp(out- out.max(axis=1, keepdims=True))
		prediction = (e_x / e_x.sum(axis=1, keepdims=True))
	print("		defining pred function...")
	pred_function = theano.function([X], prediction, allow_input_downcast=True, on_unused_input='ignore')
	return pred_function


	
def segment(valid_fn, lp, featureImages, mask):
	epochStartTime = time.clock()
	classNum = len(lp["labels"])
	sh = mask.shape
	sh_fl = featureImages[0].shape
	y = np.zeros((classNum, sh[0], sh[1], sh[2]), dtype='float32')
	featuresBatch = np.zeros((lp["batchSize"], len(lp["channels"]), sh_fl[0], sh_fl[1]), dtype='float32')
	batchIndex = 0	
	total = np.zeros(5, dtype='float32')

	caseStartTime = time.clock()
	for z in range(featureImages[0].shape[2]):
		for cc in range(len(lp["channels"])):
			featuresBatch[batchIndex, cc, :, :] = featureImages[cc][:,:,z]
		batchIndex += 1
		if batchIndex == lp["batchSize"] or z==featureImages[0].shape[2]-1:
			res = valid_fn(featuresBatch)
			for zz in range(batchIndex):
				y[:,:,:,z-batchIndex+1+zz] = res[zz,:,:,:]
			batchIndex = 0
	return y[1,:,:,:], y[2,:,:,:]





def load_parameters(network, fname):
	with np.load(fname) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	lasagne.layers.set_all_param_values(network, param_values)
	return network



def loadLearningParameters(path):
	f = open(path)
	lp = {}
	keywords = ["filterNumStart","lr", "epochs", "lambda2", "batchSize", "doBatchNorm", "channels", "continue", "dropout", "onlyPos", "lastLayer","depth", "mask", "labels", "dataPath"]
	types = ["int","listfloat", "int", "float", "int", "int", "liststring", "string", "float", "int","string","int","string", "liststring", "string"]
	found = [0]*len(keywords)
	vals = [0]*len(keywords)

	for line in f:
		if len(line) < 3:
			break
		[var, val] = line.translate(None," ")[:-1].split("=")
		for counter in range(0, len(keywords)):
			if var==keywords[counter]:
				vals[counter] = val
				found[counter] = 1
				break
	for i in range(len(keywords)):
		if found[i]==0:
			print("error: value for "+keywords[i] + " not found!")
			sys.exit(1)
		elif types[i]=='int':
			lp[keywords[i]]=int(vals[i])
		elif types[i]=='float':
			lp[keywords[i]]=float(vals[i])
		elif types[i]=='listint':
			lp[keywords[i]] = [int(x) for x in (vals[i])[1:-1].split(",")]
		elif types[i]=='listfloat':
			lp[keywords[i]] = [float(x) for x in (vals[i])[1:-1].split(",")]
		elif types[i]=='liststring':
			lp[keywords[i]] = [x for x in (vals[i])[1:-1].split(",")]
		else:
			lp[keywords[i]] = vals[i]
	
	print("================================")
	for i in range(len(keywords)):
		print(keywords[i] + ":"+ str(lp[keywords[i]]))

	return lp


def findMargins(mask):
	sh = mask.shape
	startX=0
	endX=0
	startY=0
	endY=0
	for x in range(sh[0]):
		if np.count_nonzero(mask[x:x+1,:,:])>0:
			startX = x
			break
	for x in range(sh[0]):
		if np.count_nonzero(mask[sh[0]-x-1:sh[0]-x,:,:])>0:
			endX = sh[0]-x
			break
	for y in range(sh[1]):
		if np.count_nonzero(mask[:,y:y+1,:])>0:
			startY = y
			break
	for y in range(sh[1]):
		if np.count_nonzero(mask[:,sh[1]-y-1:sh[1]-y,:])>0:
			endY = sh[1]-y
			break
	return (startX, startY, endX, endY)


def handleImage(ims, mask, cropSize, desiredSize):
	print("finding margins....")
	margins = findMargins(mask)
	print("reshaping images...", margins)
	center = [(margins[0]+margins[2])/2, (margins[1]+margins[3])/2]
	center = [max(center[0], cropSize[0]/2), max(center[1], cropSize[1]/2)]
	center = [min(ims[0].shape[0]-cropSize[0]/2,center[0]),  min(ims[0].shape[1]-cropSize[1]/2,center[1])]
	sh = ims[0].shape
	res = []
	outCenter = [desiredSize[0]/2, desiredSize[1]/2]
	for im in ims:
		out = np.zeros((desiredSize[0], desiredSize[1], sh[2]))
		out[outCenter[0]-cropSize[0]/2:outCenter[0]+cropSize[0]/2, outCenter[1]-cropSize[1]/2:outCenter[1]+cropSize[1]/2,:] = im[center[0]-cropSize[0]/2:center[0]+cropSize[0]/2,center[1]-cropSize[1]/2:center[1]+cropSize[1]/2,:]
		res.append(out)
	
	outmask = mask[center[0]-cropSize[0]/2:center[0]+cropSize[0]/2,center[1]-cropSize[1]/2:center[1]+cropSize[1]/2,:]
	return [res, outmask, center, outCenter]
		







def run_batch(data_path, cases, model_path, arch_path):
	c = cases[0]
	patient_path = data_path+c+"/"
	lp = loadLearningParameters(arch_path)
	maskImage = nib.load(patient_path+"_brainMask_fl.nii.gz")
	mask = maskImage.get_data()

	inpFileNames = ['nor_fl_biasCorrected_fl.nii.gz', 'nor_t1_biasCorrected_fl.nii.gz']
	inpFiles = []
	for i in range(len(inpFileNames)):
		inpFiles.append(nib.load(patient_path+inpFileNames[i]).get_data())
	cropSize = [356, 372]
	d = lp["depth"]
	margin = (pow(2, d-1)-1)*8+4*pow(2, d-1)
	desiredSize = [cropSize[0]+margin, cropSize[1]+margin]
	handledInps, handledMask, center, outCenter = handleImage(inpFiles, mask, cropSize, desiredSize)
	lp["tile"] = [handledInps[0].shape[0], handledInps[0].shape[1]]
	lp["outSize"] = [handledMask.shape[0], handledMask.shape[1]]
	ftensor4 = T.TensorType('float32', (False,)*4)
	X = ftensor4()
	Y = ftensor4()
	network = createUNet(X, lp)
	network = load_parameters(network, model_path)
	valid_fn = loadValidFunction(X, Y, network, lp)

	input_shape = (256,256,2)
	model = build_unet(input_shape)
	# TODO: change loss function
	model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
	model.summary()

	for c in cases:
		patient_path = data_path+c+"/"
		maskImage = nib.load(patient_path+"_brainMask_fl.nii.gz")
		mask = maskImage.get_data()
		inpFiles = []
		for i in range(len(inpFileNames)):
			inpFiles.append(nib.load(patient_path+inpFileNames[i]).get_data())
		handledInps, handledMask, center, outCenter = handleImage(inpFiles, mask, cropSize, desiredSize)

		out_c1, out_c2 = segment(valid_fn, lp, handledInps, handledMask)
		c1 = np.zeros(mask.shape, dtype="float32")
		c2 = np.zeros(mask.shape, dtype="float32")
		c1[center[0]-cropSize[0]/2:center[0]+cropSize[0]/2,center[1]-cropSize[1]/2:center[1]+cropSize[1]/2,:] = out_c1
		c2[center[0]-cropSize[0]/2:center[0]+cropSize[0]/2,center[1]-cropSize[1]/2:center[1]+cropSize[1]/2,:] = out_c2
		c1 *= mask
		c2 *= mask
		nib.save(nib.Nifti1Image(c1, maskImage.affine), patient_path+"unetd"+str(d)+"_outC1.nii.gz")
		nib.save(nib.Nifti1Image(c2, maskImage.affine), patient_path+"unetd"+str(d)+"_outC2.nii.gz")
		c1[c1<0.99] = 0
		c1[c1>=0.99] = 1
		c1 = ndimage.distance_transform_edt(1-c1)
		c2[c2<0.99] = 0
		c2[c2>=0.99] = 1
		c2 = ndimage.distance_transform_edt(1-c2)
		nib.save(nib.Nifti1Image(c1, maskImage.affine), patient_path+"dist_left_vent.nii.gz")
		nib.save(nib.Nifti1Image(c2, maskImage.affine), patient_path+"dist_right_vent.nii.gz")
		print("ventricle segmentation done..")

		

	
	
	
	


