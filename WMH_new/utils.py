from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


def iou_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def iou_coef_loss(y_true, y_pred):
    return -iou_coef(y_true, y_pred)  


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


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


def determine_training(f1, l1, f2, l2, f3, l3, case=''):
    with open('WMH_new/training_cases.txt') as f:
        if case in f.read():
            return
    with open('WMH_new/test_cases.txt') as f:
        if case in f.read():
            return
    plt.ion()
    fig, ax = plt.subplots(3,2, figsize=(10,6))
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

    plt.show()

    while True:
        user_input = input('\n Use this as training data? (y/n): ')
        if user_input.lower() == 'y':
            print(f'✓ {case}')
            with open('WMH_new/training_cases.txt', 'a') as f:
                f.write(case + '\n') 
                f.close()
            break
        elif user_input.lower() == 'n':
            print(f'✗ {case}')
            with open('WMH_new/test_cases.txt', 'a') as f:
                f.write(case + '\n') 
                f.close()
            break
        else:
            print('Type \'y\' or \'n\'')
            continue


    plt.close()