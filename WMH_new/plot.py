import matplotlib.pyplot as plt
import os
import numpy as np
import parameters
import utils
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_training(model_history, number):
    # Plot the training and validation accuracy and loss at each epoch
    # print(model_history.history)
    if 'multi' not in parameters.unet_version:
        metric = model_history.history['dice_coef']
        val_metric = model_history.history['val_dice_coef']
    else:
        metric = model_history.history['dice_coef_multilabel']
        val_metric = model_history.history['val_dice_coef_multilabel']
    plt.figure()
    plt.plot(model_history.epoch, metric, label=f'Training {parameters.training_loss[1]}')
    plt.plot(model_history.epoch, val_metric, label=f'Validation {parameters.training_loss[1]}')
    plt.title(f'Training and validation {str(parameters.training_loss[1])} #{number}')
    plt.xlabel('Epoch')
    plt.ylabel(parameters.training_loss[1])
    plt.legend()

    # Save plot
    if not os.path.exists(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(number)).replace("\\","/")):
            os.makedirs(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(number)).replace("\\","/"))
    plt.savefig(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(number), 'training_graph').replace("\\","/"))

    # plt.show()


def plot_image(image, title='Title'):
    # Plot single slice of MRI data (ndarray)
    plt.figure()
    plt.imshow(np.rot90(image), cmap='gray', origin='lower')
    plt.title(title)
    plt.show()


def plot_t1_and_fl(t1, fl):
    plt.figure()
    plt.imshow(np.rot90(t1), cmap='gray')
    plt.imshow(np.rot90(fl), cmap='hot', alpha=0.3)
    plt.show()


def plot_orig_and_lbl(t1, fl, label):
    # Plot t1, fl, lbl and prediction
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    # fig.suptitle(c)
    ax[0].imshow(np.rot90(t1), cmap='gray')
    ax[0].set_title('Original T1')
    ax[0].axis('off')

    ax[1].imshow(np.rot90(fl), cmap='gray')
    ax[1].set_title('Original FL')
    ax[1].axis('off')

    # Define the colors for each class
    colors = ['k', 'w', 'khaki', 'cornflowerblue']

    # Create a custom colormap
    cmap = ListedColormap(colors)

    im = ax[0].imshow(np.rot90(label), cmap=cmap, alpha=0.5)
    ax[0].set_title('WMH label')
    ax[0].axis('off')

    im = ax[1].imshow(np.rot90(label), cmap=cmap, alpha=0.5)
    ax[1].set_title('WMH pred')
    ax[1].axis('off')

    # Create a custom colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Background', 'WMH', 'NAWM', 'GM'])

    # Save plot
    # if not os.path.exists(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, 'predictions').replace("\\","/")):
    #         os.makedirs(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, 'predictions').replace("\\","/"))
    # plt.savefig(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, 'predictions', str(i)).replace("\\","/"))

    plt.show()


def plot_orig_and_lbls(img, wmh, nawm, gm):
    # Plot t1, fl, lbl and prediction
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    # fig.suptitle(c)
    ax[0].imshow(np.rot90(img[:,:,1]), cmap='gray')
    ax[0].set_title('FL')
    ax[0].axis('off')

    ax[1].imshow(np.rot90(img[:,:,1]), cmap='gray')
    ax[1].set_title('labels')
    ax[1].axis('off')

    ax[1].imshow(np.rot90(wmh), cmap='Oranges', alpha=0.5)

    ax[1].imshow(np.rot90(nawm), cmap='RdPu', alpha=0.5)

    ax[1].imshow(np.rot90(gm), cmap='BuGn', alpha=0.5)

    # Save plot
    # if not os.path.exists(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, 'predictions').replace("\\","/")):
    #         os.makedirs(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, 'predictions').replace("\\","/"))
    # plt.savefig(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, 'predictions', str(i)).replace("\\","/"))

    plt.show()


def plot_lbl_and_pred(t1, fl, label, pred, i=None):
    # Plot t1, fl, lbl and prediction
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    # fig.suptitle(c)
    ax[0].imshow(np.rot90(t1), cmap='gray')
    ax[0].set_title('Original T1')
    ax[0].axis('off')

    ax[1].imshow(np.rot90(fl), cmap='gray')
    ax[1].set_title('Original FL')
    ax[1].axis('off')

    ax[0].imshow(np.rot90(label), cmap='hot', alpha=0.5)
    ax[0].set_title('WMH label')
    ax[0].axis('off')

    ax[1].imshow(np.rot90(pred), cmap='hot', alpha=0.5)
    ax[1].set_title('WMH pred')
    ax[1].axis('off')

    # Save plot
    # if not os.path.exists(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, 'predictions_where').replace("\\","/")):
    #         os.makedirs(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, 'predictions_where').replace("\\","/"))
    # plt.savefig(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, 'predictions_where', str(i)).replace("\\","/"))

    # plt.show()

def plot_augmented_data(t1, fl, label, aug_t1, aug_fl, aug_label):
    # Plot t1, fl, lbl and prediction
    fig, ax = plt.subplots(2,2, figsize=(10, 10))
    # fig.suptitle(c)
    ax[0,0].imshow(np.rot90(t1), cmap='gray')
    ax[0,0].set_title('Original T1')
    ax[0,0].axis('off')

    ax[0,1].imshow(np.rot90(fl), cmap='gray')
    ax[0,1].set_title('Original FL')
    ax[0,1].axis('off')

    ax[0,0].imshow(np.rot90(label), cmap='hot', alpha=0.5)

    ax[0,1].imshow(np.rot90(label), cmap='hot', alpha=0.5)

    ax[1,0].imshow(np.rot90(aug_t1), cmap='gray')
    ax[1,0].set_title('Augmented T1')
    ax[1,0].axis('off')

    ax[1,1].imshow(np.rot90(aug_fl), cmap='gray')
    ax[1,1].set_title('Augmented FL')
    ax[1,1].axis('off')

    ax[1,0].imshow(np.rot90(aug_label), cmap='hot', alpha=0.5)

    # ax[1,1].imshow(np.rot90(aug_label), cmap='hot', alpha=0.5)

    plt.show()

def plot_prediction(fl_orig, label, wmh, nawm, gm, i=''):
    # Plot t1, fl, lbl and prediction
    fig, ax = plt.subplots(1,3, figsize=(15, 5))
    # ax[0].imshow(np.rot90(t1_orig), cmap='gray')
    # ax[0].set_title('Original T1')
    # ax[0].axis('off')

    ax[0].imshow(np.rot90(fl_orig), cmap='gray')
    ax[0].set_title('Original FL')
    ax[0].axis('off')

    ax[1].imshow(np.rot90(fl_orig), cmap='gray')

    ax[2].imshow(np.rot90(fl_orig), cmap='gray')

    # Define the colors for each class
    colors = ['k', 'darkgreen', 'steelblue', 'red']

    # Create a custom colormap
    cmap = ListedColormap(colors)

    trans_lbl = np.where(label == 0, 0, 0.5)[:,:,0]
    label_im = ax[1].imshow(np.rot90(label), alpha=np.rot90(trans_lbl), cmap=cmap)
    ax[1].set_title('WMH label')
    ax[1].axis('off')

    # divider = make_axes_locatable(ax[1])
    # cax = divider.append_axes("right", size="5%", pad=0.05)

    # Add the colorbar to the new axes
    # cbar = fig.colorbar(label_im, ax=ax[1], ticks=[0, 1, 2, 3])
    # cbar.ax.set_yticklabels(['Background', 'GM', 'NAWM', 'WMH'])
    # cbar = plt.colorbar(label_im, cax=cax)
   
    
    trans_wmh = np.where(wmh == 0, 0, 0.8)[:,:,0]
    im_wmh = ax[2].imshow(np.rot90(wmh), alpha=np.rot90(trans_wmh), cmap='Reds')
    ax[2].set_title('Prediction')
    ax[2].axis('off')

    trans_nawm = np.where(nawm == 0, 0, 0.8)[:,:,0]
    ax[2].imshow(np.rot90(nawm), alpha=np.rot90(trans_nawm), cmap='Blues')

    trans_gm = np.where(gm == 0, 0, 0.8)[:,:,0]
    ax[2].imshow(np.rot90(gm), alpha=np.rot90(trans_gm), cmap='Greens')

    # Save plot
    if not os.path.exists(os.path.join(parameters.path_pm_predictions, 'progression_v2').replace("\\","/")):
            os.makedirs(os.path.join(parameters.path_pm_predictions, 'progression_v2').replace("\\","/"))
    plt.savefig(os.path.join(parameters.path_pm_predictions, 'progression_v2', str(i)).replace("\\","/"))

    # plt.show()


def plot_pm_data(t1, fl):
    # Plot t1, fl, lbl and prediction
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    ax[0].imshow(np.rot90(t1), cmap='gray')
    ax[0].set_title('T1')
    ax[0].axis('off')

    ax[1].imshow(np.rot90(fl), cmap='gray')
    ax[1].set_title('FLAIR')
    ax[1].axis('off')

    # ax[0].imshow(np.rot90(fl), cmap='Blues', alpha=0.5)
    # ax[0].set_title('FLAIR')
    # ax[0].axis('off')

    plt.show()


def plot_pm_prediction(t1_orig, fl_orig, prediction, number, metrics=''):
    # Plot t1, fl, lbl and prediction
    fig, ax = plt.subplots(1,3, figsize=(15, 5))
    fig.suptitle(metrics)
    ax[0].imshow(np.rot90(t1_orig), cmap='gray')
    ax[0].set_title('Original T1')
    ax[0].axis('off')

    ax[1].imshow(np.rot90(fl_orig), cmap='gray')
    ax[1].set_title('Original FL')
    ax[1].axis('off')

    ax[2].imshow(np.rot90(fl_orig), cmap='gray')
    ax[2].set_title('FL with pred')
    ax[2].axis('off')

    # ax[2].imshow(np.rot90(prediction), cmap='gray')
    # ax[2].set_title('Prediction')
    # ax[2].axis('off')

    ax[2].imshow(np.rot90(prediction), cmap='hot', alpha=0.5)
    ax[2].axis('off')

    if not os.path.exists(os.path.join(parameters.path_pm_predictions, 'pm_wmh').replace("\\","/")):
            os.makedirs(os.path.join(parameters.path_pm_predictions, 'pm_wmh').replace("\\","/"))
    plt.savefig(os.path.join(parameters.path_pm_predictions, 'pm_wmh', str(number)).replace("\\","/"))
    
    # plt.show()
    plt.close()