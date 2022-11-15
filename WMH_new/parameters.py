import utils

########## U-Net parameters ##########
unet_version = 'Dice_ens_v2'                                            # !!!!! Change for new run      'IoU' ||| 'Dice' 
unet_input_shape = (200,200,2)
unet_dropout = 0.1
unet_lr = 2e-3

########## Training parameters ##########
training_batch_size = 16
training_epochs = 30
training_validation_split = 0.2
# CHANGE BOTH
training_loss = utils.dice_coef_loss, 'Dice Coeff.'               # !!!!! Change for new run     utils.iou_coef_loss, 'IoU Coeff.' ||| utils.dice_coef_loss, 'Dice Coeff.'
training_ensemble = 1

########## Datasets (train, test) ##########
path_train_img = 'WMH_new/datasets/x_train.npy'
path_train_lbl = 'WMH_new/datasets/y_train.npy'
path_test_img = 'WMH_new/datasets/x_test.npy'
path_test_lbl = 'WMH_new/datasets/y_test.npy'

########## Paths ##########
path_data_2006 = 'RUNDMC/2006'
path_data_2011 = 'RUNDMC/2011'
path_data_2015 = 'RUNDMC/2015'
atlasPath = 'WMH_new/material'

files = ['_fl_orig.nii.gz','_t1_orig.nii.gz']
label = 'wmh_semiautomatic.nii.gz'

path_2011_cases = 'RUNDMC/2011/cases.txt'
path_2015_cases = 'RUNDMC/2015/cases.txt'
path_trainingcases = 'RUNDMC/2006/train.txt'
path_validationcases = 'RUNDMC/2006/validation_new.txt.txt'
path_testcases = 'RUNDMC/2006/test50.txt'

path_model_checkpoint = 'WMH_new/checkpoints'

