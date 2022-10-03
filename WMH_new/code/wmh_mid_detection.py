import nibabel as nib
import numpy as np
import scipy.stats
import math
from scipy import ndimage



def find_margins(mask):
	x_max = y_max = 0
	startZ = endZ = -1
	startX = mask.shape[0]
	startY = mask.shape[1]
	endX=endY = 0
	for z in range(mask.shape[2]):
		count = np.count_nonzero(mask[:,:,z])
		if count:
			xs,ys,xe,ye = find_margins_slice(mask[:,:, z])
			startX = min(xs, startX)
			startY = min(ys, startY)
			endX = max(xe, endX)
			endY = max(ye, endY)
		endZ = z-1 if startZ>=0 and endZ<0 and count==0 else endZ
		startZ = z if startZ==-1 and count>0 else startZ
	endZ = mask.shape[2]-1 if endZ==-1 else endZ
	return startX, startY, startZ, endX, endY, endZ



def find_margins_slice(mask):
	sh = mask.shape
	startX=0
	endX=0
	startY=0
	endY=0
	for x in range(sh[0]):
		if np.count_nonzero(mask[x:x+1,:])>0:
			startX = x
			break
	for x in range(sh[0]):
		if np.count_nonzero(mask[sh[0]-x-1:sh[0]-x,:])>0:
			endX = sh[0]-x
			break
	for y in range(sh[1]):
		if np.count_nonzero(mask[:,y:y+1])>0:
			startY = y
			break
	for y in range(sh[1]):
		if np.count_nonzero(mask[:,sh[1]-y-1:sh[1]-y])>0:
			endY = sh[1]-y
			break
	return startX, startY, endX, endY


def find_midsagittal(t1, mask):
	curve_penalty = 100
	out_of_mask_penalty=100
	[startX, startY, startZ, endX, endY, endZ] = find_margins(mask)
	midx = (startX+endX)/2
	w = (endX-startX)/3
	h = endY-startY+1

	ventStart = int(startZ+(endZ-startZ)*0.8)
	#print w, h, ventStart
	result = np.zeros(t1.shape, dtype="float32")
	gaussian = scipy.stats.norm((startY+endY)/2,math.sqrt(h))
	coef = np.transpose(np.tile(np.transpose(np.asarray([[gaussian.pdf(x) for x in range(startY, endY+1)]])), (1,w))*1000000.0)
	cost = np.zeros((w, h, endZ-startZ+1), dtype="float32")
	dirs = np.zeros((w, h, endZ-startZ+1), dtype="int32")
	#print startZ, endZ, ventStart,"++++++++++++++"
	for zz in range(endZ+1-startZ):
		z = endZ-zz
		#print z
		data = (t1[:,:,z]+(1-mask[:,:,z])*out_of_mask_penalty)[midx-w/2:midx+int(math.ceil(w/2.0)), startY:endY+1]
		curZ = z-startZ
		if z==ventStart:
			penaltyImage = ndimage.distance_transform_edt(1-result[midx-w/2:midx+int(math.ceil(w/2.0)), startY:endY+1, ventStart+1])
		if z<=ventStart:
			#print data.shape, coef.shape, penaltyImage.shape
			data += coef*penaltyImage
		cost[w/2,0] = data[w/2,0]
		dirs[w/2,0, curZ] = 2
		for i in range(1, int(math.ceil(w/2.0))):
			cost[w/2+i,0,curZ] = cost[w/2+i-1,0,curZ]+data[w/2+i, 0]
			dirs[w/2+i,0,curZ] = 3
			cost[w/2-i,0,curZ] = cost[w/2-i+1,0,curZ]+data[w/2-i, 0]
			dirs[w/2-i,0,curZ] = 1

		for i in range(1, h):
			for j in range(w):
				left = 9999999 if j==0 else cost[j-1, i-1, curZ]+data[j, i]+curve_penalty
				middle = cost[j, i-1, curZ]+data[j, i]
				right = 9999999 if j==w-1 else cost[j+1, i-1, curZ]+data[j, i]+curve_penalty
				if left<middle and left<right:
					cost[j, i, curZ] = left
					dirs[j, i, curZ] = 1
				elif right<left and right<middle:
					cost[j, i, curZ] = right
					dirs[j, i, curZ] = 3
				else:
					cost[j, i, curZ] = middle
					dirs[j, i, curZ] = 2
					
		x = startX+3*w/2
		y = startY+h-1
		ss = dirs[:,:,curZ].shape
		for i in range(h):
			result[x, y-i, z] = 1
			x -= 2-dirs[x-startX-w, y-i-startY, curZ]
		#plt.matshow(dirs[:,:,curZ].transpose(), cmap=cm.gray)
		#plt.show()

	return result
			
		
def run(data_path, c):	

	t1 = nib.load(data_path+c+"/_t1_biasCorrected_fl.nii.gz").get_data()
	mask = nib.load(data_path+c+"/_brainMask_fl.nii.gz")
	mids = find_midsagittal(t1, mask.get_data())
	nib.save(nib.Nifti1Image(mids, mask.affine), data_path+c+"/midsagittal.nii.gz")
	distImage = ndimage.distance_transform_edt(1-mids)
	nib.save(nib.Nifti1Image(distImage, mask.affine), data_path+c+"/dist_midsagitall.nii.gz")
	
