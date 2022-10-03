### U-Net parameters
unet_dropout = 0.1
unet_lr = 1e-3

### Paths
path_data = 'WMH_old/train_and_test/test_images'
atlasPath = 'WMH_old/material/atlas'

files = ['_fl_orig.nii.gz','_t1_orig.nii.gz']
label = 'WMH_exp1.nii.gz'

path_trainingcases = 'WMH_old/train_and_test/test_images/train.txt'
path_validationcases = 'WMH_old/train_and_test/test_images/validation.txt'
path_testcases = 'WMH_old/train_and_test/test_images/test.txt'

