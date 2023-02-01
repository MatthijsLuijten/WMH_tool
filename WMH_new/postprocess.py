import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = '1'

def run(test_img, preds):
    # Use FL and T1 as inclusion mask
    preds = inclusion_mask(test_img, preds)
    
    # Create masks for regions
    wmh_mask, nawm_mask, gm_mask = create_masks(test_img, preds)
    
    # Create transition zones
    wmh_zones = create_transition_zones(wmh_mask)
    nawm_zones = create_transition_zones(nawm_mask)

    return preds, wmh_zones, nawm_zones, gm_mask


def inclusion_mask(test_img, preds):
    # Use FL and T1 as inclusion mask
    t1_mask = np.where(test_img[:,:,:,0] == 0, 0, 1)
    t1_mask = np.repeat(t1_mask[:, :, :, np.newaxis], preds.shape[-1], axis=-1)
    fl_mask = np.where(test_img[:,:,:,1] == 0, 0, 1)
    fl_mask = np.repeat(fl_mask[:, :, :, np.newaxis], preds.shape[-1], axis=-1)
    preds = np.multiply(preds, fl_mask)
    preds = np.multiply(preds, t1_mask)
    return preds


def create_masks(test_img, preds):
    test_img2 = np.expand_dims(test_img, axis=-1)
    wmh_mask = np.where(preds == 1, test_img2[:,:,:,1], 0)
    nawm_mask = np.where(preds == 2, test_img2[:,:,:,1], 0)
    gm_mask = np.where(preds == 3, 1, 0)
    return wmh_mask, nawm_mask, gm_mask


def create_transition_zones(mask):
    transition_zones = []
    for i in range(len(mask)):
        # extract one image from the 4D array
        region = mask[i][..., 0]

        # get indices of non-zero (WMH/NAWM) pixels
        nonzero_idx = np.nonzero(region)

        # get values of non-zero pixels
        nonzero_vals = region[nonzero_idx]

        # cluster the non-zero pixels into 3 classes using KMeans
        kmeans = KMeans(n_clusters=3)
        labels = kmeans.fit_predict(nonzero_vals.reshape(-1,1)) + 1 # add 1 to the labels to avoid using 0 as a cluster label

        # calculate mean intensity for each cluster
        mean_intensities = []
        for j in range(1, 4):
            indices = np.where(labels == j)
            mean_intensities.append(np.mean(nonzero_vals[indices]))

        # sort the clusters based on mean intensity
        sorted_indices = np.argsort(mean_intensities)

        # update the labels with the sorted indices
        sorted_labels = np.zeros_like(labels)
        for j, idx in enumerate(sorted_indices):
            sorted_labels[labels == idx + 1] = j + 1

        # create 2D label array for this image
        labels_2d = np.zeros_like(region)
        labels_2d[nonzero_idx] = sorted_labels # fill in the WMH pixels with cluster labels

        # set all background pixels to 0
        labels_2d[labels_2d == 0] = 0

        # add this 2D label array to the list of transition zones
        transition_zones.append(labels_2d.reshape(*region.shape, 1))
                
    return transition_zones
