### U-Net parameters
unet_dropout = 0.1
unet_lr = 1e-3

### Paths
path_data = 'RUNDMC/2006'
atlasPath = 'WMH_old/material/atlas'

files = ['_fl_orig.nii.gz','_t1_orig.nii.gz']
label = 'WMH_exp1.nii.gz'

path_trainingcases = 'RUNDMC/2006/train.txt'
path_validationcases = 'RUNDMC/2006/validation_new.txt.txt'
path_testcases = 'RUNDMC/2006/test50.txt'

# path_trainingcases = 'WMH_old/test_images/train.txt'
# path_validationcases = 'WMH_old/test_images/validation.txt'
# path_testcases = 'WMH_old/test_images/test.txt'

