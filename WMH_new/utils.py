from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import parameters
from tqdm import tqdm
import albumentations as A
from sklearn.utils import shuffle

def dice_coef(y_true, y_pred):
    """
    Dice coefficient metric for image segmentation models.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    """
    Dice coefficient loss function for image segmentation models.
    """
    return -dice_coef(y_true, y_pred)



def dice_coef_multilabel(y_true, y_pred, num_labels=4):
    """
    Dice coefficient metric for multilabel image segmentation models.
    """
    weights = [0.74398218, 1.334466, 0.64021947, 2.9023027]
    dice = 0.0
    for index in range(num_labels):
        dice += (dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index]) * weights[index])
    return dice / sum(weights)


def dice_coef_multilabel_loss(y_true, y_pred):
    """
    Dice coefficient loss function for multilabel image segmentation models.
    """
    return -dice_coef_multilabel(y_true, y_pred)



def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)


def determine_training(f1, l1, f2, l2, f3, l3, f4, l4, case=''):
    # Check if case is in any of the existing text files, return if so
    for filename in ['good.txt', 'bad.txt', 'maybe.txt']:
        with open(f'WMH_new/{filename}') as f:
            if case in f.read():
                return
    plt.ion()
    fig, ax = plt.subplots(4,2, figsize=(7,15))
    fig.suptitle(case)
    ax[0,0].imshow(np.rot90(f1), cmap='gray')
    ax[0,0].set_title('Original FL')
    ax[0,0].axis('off')

    ax[0,1].imshow(np.rot90(f1), cmap='gray')
    ax[0,1].set_title('FL and LBL')
    ax[0,1].axis('off')

    # ax[0].imshow(np.rot90(label), cmap='hot', alpha=0.5)
    # ax[0].set_title('WMH label')
    # ax[0].axis('off')

    ax[0,1].imshow(np.rot90(l1), cmap='hot', alpha=0.5)
    # ax[1].set_title('WMH label')
    ax[0,1].axis('off')

    ax[1,0].imshow(np.rot90(f2), cmap='gray')
    ax[1,0].set_title('Original FL')
    ax[1,0].axis('off')

    ax[1,1].imshow(np.rot90(f2), cmap='gray')
    ax[1,1].set_title('FL and LBL')
    ax[1,1].axis('off')

    ax[1,1].imshow(np.rot90(l2), cmap='hot', alpha=0.5)
    # ax[1].set_title('WMH label')
    ax[1,1].axis('off')

    ax[2,0].imshow(np.rot90(f3), cmap='gray')
    ax[2,0].set_title('Original FL')
    ax[2,0].axis('off')

    ax[2,1].imshow(np.rot90(f3), cmap='gray')
    ax[2,1].set_title('FL and LBL')
    ax[2,1].axis('off')

    ax[2,1].imshow(np.rot90(l3), cmap='hot', alpha=0.5)
    # ax[1].set_title('WMH label')
    ax[2,1].axis('off')

    ax[3,0].imshow(np.rot90(f4), cmap='gray')
    ax[3,0].set_title('Original FL')
    ax[3,0].axis('off')

    ax[3,1].imshow(np.rot90(f4), cmap='gray')
    ax[3,1].set_title('FL and LBL')
    ax[3,1].axis('off')

    ax[3,1].imshow(np.rot90(l4), cmap='hot', alpha=0.5)
    # ax[1].set_title('WMH label')
    ax[3,1].axis('off')

    plt.show()

    while True:
        user_input = input('\n Use this as training data? (y/n/m): ')
        if user_input.lower() == 'y':
            print(f'✓ {case}')
            with open('WMH_new/good.txt', 'a') as f:
                f.write(case + '\n') 
                f.close()
            break
        elif user_input.lower() == 'n':
            print(f'✗ {case}')
            with open('WMH_new/bad.txt', 'a') as f:
                f.write(case + '\n') 
                f.close()
            break
        elif user_input.lower() == 'm':
            print(f'✓✗ {case}')
            with open('WMH_new/maybe.txt', 'a') as f:
                f.write(case + '\n') 
                f.close()
            break
        else:
            print('Type \'y\' or \'n\' or \'m\'')
            continue

    plt.close()

def original_or_flirt(f1, orig, flirt, case=''):
    plt.ion()
    fig, ax = plt.subplots(1,3, figsize=(15, 5))

    ax[0].imshow(np.rot90(f1), cmap='gray')
    ax[0].set_title('Original FL')
    ax[0].axis('off')

    ax[1].imshow(np.rot90(f1), cmap='gray')
    ax[1].imshow(np.rot90(orig), cmap='hot', alpha=0.5)
    ax[1].set_title('Original LFB')
    ax[1].axis('off')

    ax[2].imshow(np.rot90(f1), cmap='gray')
    ax[2].imshow(np.rot90(flirt), cmap='hot', alpha=0.5)
    ax[2].set_title('FLIRTED LFB')
    ax[2].axis('off')

    plt.show()

    while True:
        user_input = input('\n Left or right or not? ')
        if user_input.lower() == 'l':
            plt.close()
            return orig
        elif user_input.lower() == 'r':
            plt.close()
            return flirt
        elif user_input.lower() == 'n':
            plt.close()
            return np.zeros_like(orig)
        else:
            print('Type \'l\' or \'r\'')
            continue



def save_datasets(train_img, train_lbl, test_img, test_lbl):
    # Save datasets
    np.save(parameters.path_train_img, train_img)
    np.save(parameters.path_train_lbl, train_lbl)
    np.save(parameters.path_test_img, test_img)
    np.save(parameters.path_test_lbl, test_lbl)


def save_pm_datasets(train_img, train_lbl_wmh, train_lbl_nawm, train_lbl_gm, test_img, test_lbl_wmh, test_lbl_nawm, test_lbl_gm):
    # Save datasets
    # PM WMH INPUT IMAGES
    np.save(parameters.path_pm_train_img_v2, train_img)
    np.save(parameters.path_pm_test_img_v2, test_img)

    # PM WMH
    np.save(parameters.path_wmh_train_lbl_v2, train_lbl_wmh)
    np.save(parameters.path_wmh_test_lbl_v2, test_lbl_wmh)

    # PM NAWM
    np.save(parameters.path_nawm_train_lbl_v2, train_lbl_nawm)
    np.save(parameters.path_nawm_test_lbl_v2, test_lbl_nawm)

    # PM GM
    np.save(parameters.path_gm_train_lbl_v2, train_lbl_gm)
    np.save(parameters.path_gm_test_lbl_v2, test_lbl_gm)


def normalize_image(image):
    min_val, max_val = np.min(image), np.max(image)
    range = max_val - min_val
    if range == 0:
        return np.array(image)
    else:
        return (image - min_val) / range


def augment_data(train_img, train_lbl):
    aug = A.Compose([
        A.VerticalFlip(p=0.5),              
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, border_mode=0, value=0., p=.5),
            A.GridDistortion(p=0.5, border_mode=0, value=0.),
            ], p=0.8),
        A.GaussNoise(var_limit=(0.1),p=.3),
        A.RandomBrightnessContrast(p=0.8),    
        A.RandomGamma(p=0.8)],
        p=.95)
    
    aug_train_img, aug_train_lbl = [], []
    for i, t in tqdm(enumerate(train_img)):
        for _ in range(41):
            augmented = aug(image=t, mask=train_lbl[i])
            augmented_input = augmented['image']
            augmented_mask = augmented['mask']
            aug_train_img.append(augmented_input)
            aug_train_lbl.append(augmented_mask)

    return shuffle(np.array(aug_train_img), np.array(aug_train_lbl))
