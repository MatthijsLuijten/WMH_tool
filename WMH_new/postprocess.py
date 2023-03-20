import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
import csv
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = '1'
from parameters import *

def run(mri, preds, lfb, cases):
    # Use FL and T1 as inclusion mask
    preds = inclusion_mask(mri, preds)
    
    # Create masks for regions
    wmh_mask, nawm_mask, gm_mask = create_masks(mri, preds)
    
    # Create transition zones
    wmh_zones = create_transition_zones(wmh_mask)
    nawm_zones = create_transition_zones(nawm_mask)

    # Perform correlation analysis between zones and LFB
    # visualize_correlation(wmh_zones, nawm_zones, gm_mask, lfb, cases)
    correlation(wmh_zones, nawm_zones, gm_mask, lfb, cases, mri)

    return preds, wmh_zones, nawm_zones, gm_mask, lfb


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
    wmh_mask = np.where(preds == 3, test_img2[:,:,:,1], 0)
    nawm_mask = np.where(preds == 2, test_img2[:,:,:,1], 0)
    gm_mask = np.where(preds == 1, 1, 0)
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

        if len(nonzero_vals) == 0:
            labels = np.array([])
        else:
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

def visualize_correlation(wmh_zone, nawm_zone, lfbs):
    all_zone_values = []
    all_lfb_values = []

    all_wmh_zone_values = []
    all_wmh_lfb_values = []

    all_nawm_zone_values = []
    all_nawm_lfb_values = []
    
    # Loop through cases
    for i, (zone, lfb) in enumerate(zip(wmh_zone, lfbs)):
        zone = zone[:, :, 0]
        
        # check if zone consists only of 0 values
        if np.count_nonzero(zone) == 0:
            continue
        
        non_zero_indices = np.nonzero(zone)
        zone_values = zone[non_zero_indices]
        lfb_values = lfb[non_zero_indices]
        
        non_zero_lfb_indices = np.nonzero(lfb_values)
        lfb_values = lfb_values[non_zero_lfb_indices]
        zone_values = zone_values[non_zero_lfb_indices]
        
        all_zone_values.append(zone_values)
        all_lfb_values.append(lfb_values)
        all_wmh_zone_values.append(zone_values)
        all_wmh_lfb_values.append(lfb_values)
    
    all_wmh_zone_values = np.concatenate(all_wmh_zone_values)
    all_wmh_lfb_values = np.concatenate(all_wmh_lfb_values)
    r, p = spearmanr(all_wmh_zone_values, all_wmh_lfb_values)
    print("Pearson correlation coefficient for WMH: ", r)
    print("p-value for WMH: ", p)

    # Plot boxplots for all zones together
    plt.figure(figsize=(4, 4))
    for zone_value in [1, 2, 3]:
        indices = np.where(all_wmh_zone_values == zone_value)
        zone_data = all_wmh_lfb_values[indices]
        plt.boxplot(zone_data, positions=[zone_value], showfliers=False)

    plt.xlabel('WMH Severity')
    plt.ylabel('LFB Pixel Intensity (meyelin loss)')
    plt.title('LFB Intensity for each Zone')
    plt.xticks([1, 2, 3], ['Mild', 'Moderate', 'Severe'])
    plt.show()
    
    for i, (zone, lfb) in enumerate(zip(nawm_zone, lfbs)):
        zone = zone[:, :, 0]
        
        # check if zone consists only of 0 values
        if np.count_nonzero(zone) == 0:
            continue
        
        non_zero_indices = np.nonzero(zone)
        zone_values = zone[non_zero_indices]
        lfb_values = lfb[non_zero_indices]
        
        non_zero_lfb_indices = np.nonzero(lfb_values)
        lfb_values = lfb_values[non_zero_lfb_indices]
        zone_values = zone_values[non_zero_lfb_indices]
        
        all_zone_values.append(zone_values)
        all_lfb_values.append(lfb_values)
        all_nawm_zone_values.append(zone_values)
        all_nawm_lfb_values.append(lfb_values)
        
    all_nawm_zone_values = np.concatenate(all_nawm_zone_values)
    all_nawm_lfb_values = np.concatenate(all_nawm_lfb_values)
    r, p = spearmanr(all_nawm_zone_values, all_nawm_lfb_values)
    print("Pearson correlation coefficient for NAWM: ", r)
    print("p-value for NAWM: ", p)

    # Plot boxplots for all zones together
    plt.figure(figsize=(4, 4))
    for zone_value in [1, 2, 3]:
        indices = np.where(all_nawm_zone_values == zone_value)
        zone_data = all_nawm_lfb_values[indices]
        plt.boxplot(zone_data, positions=[zone_value], showfliers=False)

    plt.xlabel('NAWM Severity')
    plt.ylabel('LFB Pixel Intensity (meyelin loss)')
    plt.title('LFB Intensity for each Zone')
    plt.xticks([1, 2, 3], ['Mild', 'Moderate', 'Severe'])
    plt.show()
    
    all_zone_values_combined = all_wmh_zone_values.tolist() + all_nawm_zone_values.tolist()
    all_lfb_values_combined = all_wmh_lfb_values.tolist() + all_nawm_lfb_values.tolist()

    # Calculate Pearson correlation coefficient for combined data
    r, p = spearmanr(all_zone_values_combined, all_lfb_values_combined)
    print("Pearson correlation coefficient for combined WMH and NAWM data: ", r)
    print("p-value for combined WMH and NAWM data: ", p)

    # Plot boxplots for WMH and NAWM together
    plt.figure(figsize=(6, 4))
    data = [all_wmh_lfb_values, all_nawm_lfb_values]
    labels = ['WMH', 'NAWM']
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel('LFB Pixel Intensity (meyelin loss)')
    plt.title('LFB Intensity for WMH and NAWM')
    plt.show()


def correlation(wmh_zones, nawm_zones, gm_mask, lfbs, cases, mri):
    # Loop over each case
    for case_id in range(len(cases)):
        # Get the corresponding WMH and NAWM zones
        wmh_zone = wmh_zones[case_id]
        nawm_zone = nawm_zones[case_id]
        gm_zone = gm_mask[case_id]
        lfb_zone = lfbs[case_id]
        flair = mri[case_id,:,:,1]

        wmh = np.ndarray.flatten(wmh_zone)
        nawm = np.ndarray.flatten(nawm_zone)
        gm = np.ndarray.flatten(gm_zone)
        lfb = np.ndarray.flatten(lfb_zone)
        flair = np.ndarray.flatten(flair)

        output_file = os.path.join(path_pm_wmh_data, cases[case_id], 'correlation.csv')
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['case_id', 'pixel_num', 'ROI', 'cluster', 'FLAIR', 'LFB'])
            # Loop over each pixel in the image
            for i in range(40000):
                if wmh[i] != 0:
                    writer.writerow([cases[case_id], i+1, 'WMH', wmh[i], flair[i], lfb[i]])
                elif nawm[i] != 0:
                    writer.writerow([cases[case_id], i+1, 'NAWM', nawm[i], flair[i], lfb[i]])
                elif gm[i] != 0:
                    writer.writerow([cases[case_id], i+1, 'GM', gm[i], flair[i], lfb[i]])
                else:
                    writer.writerow([cases[case_id], i+1, 'Background', 0, flair[i], lfb[i]])
            


def run_inference(test_img, preds):
    # Use FL and T1 as inclusion mask
    preds = inclusion_mask(test_img, preds)
    
    # Create masks for regions
    wmh_mask, nawm_mask, gm_mask = create_masks(test_img, preds)
    
    # Create transition zones
    wmh_zones = create_transition_zones(wmh_mask)
    nawm_zones = create_transition_zones(nawm_mask)

    return preds, wmh_zones, nawm_zones, gm_mask