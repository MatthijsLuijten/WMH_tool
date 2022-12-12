import matplotlib.pyplot as plt
import os
import numpy as np
import parameters
import utils

def plot_training(model_history, number):
    # Plot the training and validation accuracy and loss at each epoch
    # print(model_history.history)
    if parameters.training_loss[0] == utils.dice_coef_loss:
        metric = model_history.history['dice_coef']
        val_metric = model_history.history['val_dice_coef']
    elif parameters.training_loss[0] == utils.iou_coef_loss:
        metric = model_history.history['iou_coef']
        val_metric = model_history.history['val_iou_coef']
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


def plot_orig_and_lbl(t1, fl, label):
    # Plot t1, fl, lbl and prediction
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    ax[0].imshow(np.rot90(t1), cmap='gray')
    ax[0].set_title('Original T1')
    ax[0].axis('off')

    ax[1].imshow(np.rot90(fl), cmap='gray')
    ax[1].set_title('Original FL')
    ax[1].axis('off')

    ax[0].imshow(np.rot90(label), cmap='hot', alpha=0.5)
    ax[0].set_title('WMH label')
    ax[0].axis('off')

    ax[1].imshow(np.rot90(label), cmap='hot', alpha=0.5)
    ax[1].set_title('WMH label')
    ax[1].axis('off')

    plt.show()


def plot_prediction(t1_orig, fl_orig, label, prediction, metrics=''):
    # Plot t1, fl, lbl and prediction
    fig, ax = plt.subplots(1,4, figsize=(15, 5))
    fig.suptitle(metrics)
    ax[0].imshow(np.rot90(t1_orig), cmap='gray')
    ax[0].set_title('Original T1')
    ax[0].axis('off')

    ax[1].imshow(np.rot90(fl_orig), cmap='gray')
    ax[1].set_title('Original FL')
    ax[1].axis('off')

    ax[2].imshow(np.rot90(label), cmap='gray')
    ax[2].set_title('WMH label')
    ax[2].axis('off')

    ax[3].imshow(np.rot90(prediction), cmap='gray')
    ax[3].set_title('Prediction')
    ax[3].axis('off')

    # Save plot
    if not os.path.exists(os.path.join(parameters.path_model_checkpoint, parameters.unet_version).replace("\\","/")):
            os.makedirs(os.path.join(parameters.path_model_checkpoint, parameters.unet_version).replace("\\","/"))
    plt.savefig(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, 'prediction').replace("\\","/"))

    plt.show()


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