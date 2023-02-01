import utils

########## U-Net parameters ##########
unet_version = 'pm_multi'                                           
unet_input_shape = (200,200,2)
unet_dropout = 0.1
unet_lr = 2e-3

########## Training parameters ##########
training_batch_size = 16
training_epochs = 100
training_validation_split = 0.2
training_loss = utils.dice_coef_loss, 'Dice Coeff.'
training_ensemble = 3

########## Datasets (train, test) ##########
path_train_img = 'WMH_new/datasets/x_182x218_train.npy'
path_train_lbl = 'WMH_new/datasets/y_182x218_train.npy'
path_test_img = 'WMH_new/datasets/x_182x218_test.npy'
path_test_lbl = 'WMH_new/datasets/y_182x218_test.npy'

# PM WMH INPUT IMAGES
path_pm_train_img = 'WMH_new/datasets/pm_x_200x200_train.npy'
path_pm_test_img = 'WMH_new/datasets/pm_x_200x200_test.npy'

# PM WMH
path_wmh_train_lbl = 'WMH_new/datasets/wmh_train_lbl.npy'
path_wmh_test_lbl = 'WMH_new/datasets/wmh_test_lbl.npy'

# PM NAWM
path_nawm_train_lbl = 'WMH_new/datasets/nawm_train_lbl.npy'
path_nawm_test_lbl = 'WMH_new/datasets/nawm_test_lbl.npy'

# PM GM
path_gm_train_lbl = 'WMH_new/datasets/gm_train_lbl.npy'
path_gm_test_lbl = 'WMH_new/datasets/gm_test_lbl.npy'

# PM Multi Class
path_pm_train_img_classes = 'WMH_new/datasets/pm_x_200x200_train_classes.npy'
path_pm_train_lbl_classes = 'WMH_new/datasets/pm_y_200x200_train_classes.npy'
path_pm_test_img_classes = 'WMH_new/datasets/pm_x_200x200_test_classes.npy'
path_pm_test_lbl_classes = 'WMH_new/datasets/pm_y_200x200_test_classes.npy'

########## Paths ##########
path_data_2006 = 'RUNDMC/2006'
path_data_2011 = 'RUNDMC/2011'
path_data_2015 = 'RUNDMC/2015'
atlasPath = 'WMH_new/material'

files = ['_fl_orig.nii.gz','_t1_orig.nii.gz']
label = 'wmh_semiautomatic.nii.gz'

path_good_cases = 'WMH_new/material/good.txt'
path_bad_cases = 'WMH_new/material/maybe.txt'
path_2011_cases = 'RUNDMC/2011/cases.txt'
path_2015_cases = 'RUNDMC/2015/cases.txt'
path_trainingcases = 'RUNDMC/2006/train.txt'
path_validationcases = 'RUNDMC/2006/validation_new.txt.txt'
path_testcases = 'RUNDMC/2006/test50.txt'

path_model_checkpoint = 'WMH_new/checkpoints'

########## Post Mortem ##########
path_pm_mri_cases = 'E:/Matthijs/postmortem_MRI/cases.txt'
path_pm_mri_data = 'E:/Matthijs/postmortem_MRI'
path_pm_mri_dataset = 'WMH_new/datasets/pm_mri_data.npy'

path_pm_wmh_cases = 'E:/Matthijs/postmortem_WMH/cases.txt'
path_pm_wmh_data = 'E:/Matthijs/postmortem_WMH'
path_pm_wmh_dataset = 'WMH_new/datasets/pm_wmh_data.npy'


path_pm_predictions = 'WMH_new/pm_predictions'