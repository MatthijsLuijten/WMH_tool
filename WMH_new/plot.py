import matplotlib.pyplot as plt
import os
import numpy as np
import parameters
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap


def plot_training(model_history, number):
    # Plot the training and validation accuracy and loss at each epoch
    # print(model_history.history)
    metric = model_history.history['dice_coef_multilabel']
    val_metric = model_history.history['val_dice_coef_multilabel']
    plt.figure()
    plt.plot(model_history.epoch, metric, label=f'Training Dice Coefficient')
    plt.plot(model_history.epoch, val_metric, label=f'Validation Dice Coefficient')
    plt.title(f'Training and validation Dice Coefficient #{number}')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
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


def plot_lbl_and_pred(fl, label, pred, i=None):
    cols = ['Original FLAIR', 'WMH label', 'WMH prediction']
    rows = ['Slice 70','Slice 80','Slice 90','Slice 100','Slice 110']
    # Plot t1, fl, lbl and prediction
    fig, ax = plt.subplots(5,3)
    for axes, col in zip(ax[0], cols):
        axes.set_title(col)

    for axes, row in zip(ax[:,0], rows):
        axes.set_ylabel(row, rotation=90, size='large')

    fl1, lbl1, pred1 = fl[:,:,70], label[:,:,70], pred[:,:,70]
    fl2, lbl2, pred2 = fl[:,:,80], label[:,:,80], pred[:,:,80]
    fl3, lbl3, pred3 = fl[:,:,90], label[:,:,90], pred[:,:,90]
    fl4, lbl4, pred4 = fl[:,:,100], label[:,:,100], pred[:,:,100]
    fl5, lbl5, pred5 = fl[:,:,110], label[:,:,110], pred[:,:,110]
    
    colors = [(0, 0, 0), (1, 0, 0)] # first color is black, last is red
    cm = LinearSegmentedColormap.from_list(
            "Custom", colors, N=20)

    # First row
    ax[0, 0].imshow(np.rot90(fl1), cmap='gray')

    trans_label = np.where(lbl1 == 0, 0, 0.8)[:,:]
    ax[0, 1].imshow(np.rot90(lbl1), cmap=cm)#, alpha=np.rot90(trans_label))
    ax[0, 1].set_facecolor('black')
        
    trans_pred = np.where(pred1 == 0, 0, 0.8)[:,:,0]
    ax[0, 2].imshow(np.rot90(pred1), cmap=cm)#, alpha=np.rot90(trans_pred))

    # Second row
    ax[1, 0].imshow(np.rot90(fl2), cmap='gray')

    trans_label = np.where(lbl2 == 0, 0, 0.8)[:,:]
    ax[1, 1].imshow(np.rot90(lbl2), cmap='Reds', alpha=np.rot90(trans_label))
        
    trans_pred = np.where(pred2 == 0, 0, 0.8)[:,:,0]
    ax[1, 2].imshow(np.rot90(pred2), cmap='Reds', alpha=np.rot90(trans_pred))

    # Third row
    ax[2, 0].imshow(np.rot90(fl3), cmap='gray')

    trans_label = np.where(lbl3 == 0, 0, 0.8)[:,:]
    ax[2, 1].imshow(np.rot90(lbl3), cmap='Reds', alpha=np.rot90(trans_label))
        
    trans_pred = np.where(pred3 == 0, 0, 0.8)[:,:,0]
    ax[2, 2].imshow(np.rot90(pred3), cmap='Reds', alpha=np.rot90(trans_pred))

    # Fourth row
    ax[3, 0].imshow(np.rot90(fl4), cmap='gray')

    trans_label = np.where(lbl4 == 0, 0, 0.8)[:,:]
    ax[3, 1].imshow(np.rot90(lbl4), cmap='Reds', alpha=np.rot90(trans_label))
        
    trans_pred = np.where(pred4 == 0, 0, 0.8)[:,:,0]
    ax[3, 2].imshow(np.rot90(pred4), cmap='Reds', alpha=np.rot90(trans_pred))

    # Fifth row
    ax[4, 0].imshow(np.rot90(fl5), cmap='gray')

    trans_label = np.where(lbl5 == 0, 0, 0.8)[:,:]
    ax[4, 1].imshow(np.rot90(lbl5), cmap='Reds', alpha=np.rot90(trans_label))
        
    trans_pred = np.where(pred5 == 0, 0, 0.8)[:,:,0]
    ax[4, 2].imshow(np.rot90(pred5), cmap='Reds', alpha=np.rot90(trans_pred))

    for i in range(5):
        for j in range(3):
            ax[i, j].axis('off')
    fig.tight_layout()

    plt.show()

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

def plot_prediction(fl_orig, label, wmh, nawm, gm, lfb, i=''):
    # Plot t1, fl, lbl and prediction
    fig, ax = plt.subplots(1,3, figsize=(15, 5))

    ax[0].imshow(np.rot90(fl_orig), cmap='gray')
    ax[0].set_title('Original FL')
    ax[0].axis('off')

    ax[1].imshow(np.rot90(fl_orig), cmap='gray')
    ax[1].imshow(np.rot90(lfb), cmap='hot', alpha=0.5)

    ax[2].imshow(np.rot90(fl_orig), cmap='gray')

    # Define the colors for each class
    colors = ['k', 'darkgreen', 'steelblue', 'red']

    # Create a custom colormap
    cmap = ListedColormap(colors)

    # trans_lbl = np.where(label == 0, 0, 0.5)[:,:,0]
    # label_im = ax[1].imshow(np.rot90(label), alpha=np.rot90(trans_lbl), cmap=cmap)
    # ax[1].set_title('WMH label')
    # ax[1].axis('off')
   
    trans_nawm = np.where(nawm == 0, 0, 0.8)[:,:,0]
    ax[2].imshow(np.rot90(nawm), alpha=np.rot90(trans_nawm), cmap='Blues')

    trans_gm = np.where(gm == 0, 0, 0.8)[:,:,0]
    ax[2].imshow(np.rot90(gm), alpha=np.rot90(trans_gm), cmap='Greens')

    trans_wmh = np.where(wmh == 0, 0, 0.8)[:,:,0]
    im_wmh = ax[2].imshow(np.rot90(wmh), alpha=np.rot90(trans_wmh), cmap='Reds')
    ax[2].set_title('Prediction')
    ax[2].axis('off')

    # Save plot
    # if not os.path.exists(os.path.join(parameters.path_pm_predictions, 'progression_v4').replace("\\","/")):
    #         os.makedirs(os.path.join(parameters.path_pm_predictions, 'progression_v4').replace("\\","/"))
    # plt.savefig(os.path.join(parameters.path_pm_predictions, 'progression_v4', str(i)).replace("\\","/"))
    plt.title(str(i))
    plt.show()


def plot_prediction_inference(fl_orig, wmh, nawm, gm, i, cases):
    wmh = np.where(wmh == 0, 0, np.where(wmh == 1, 5, np.where(wmh == 2, 6, 7)))
    nawm = np.where(nawm == 0, 0, np.where(nawm == 1, 2, np.where(nawm == 2, 3, 4)))
    total = wmh+nawm+gm

    # Plot prediction
    fig, ax = plt.subplots(figsize=(2,2))

    # Create a custom colormap
    colors = plt.cm.jet(np.linspace(0, 1, 256))
    newcolors = colors[1:,:]
    cmap = ListedColormap(newcolors)

    alpha = np.where(total == 0, 0, 1.0)[:,:,0]

    ax.imshow(total, alpha=alpha, cmap=cmap, vmin=1, vmax=7)
    ax.axis('off')
    fig.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    # Save plot
    #Change folder where output will be saved! so 'C338C_MRI' to NEW!
    if not os.path.exists(os.path.join(parameters.path_pm_predictions, 'C338C_MRI', cases[i]).replace("\\","/")):
            os.makedirs(os.path.join(parameters.path_pm_predictions, 'C338C_MRI', cases[i]).replace("\\","/"))
    plt.savefig(os.path.join(parameters.path_pm_predictions, 'C338C_MRI', cases[i], cases[i]+'_total').replace("\\","/"))
    plt.close()
    # Plot prediction
    fig, ax = plt.subplots(figsize=(2,2))
    ax.imshow(fl_orig, cmap='gray')
    alpha = np.where(total == 0, 0, 0.8)[:,:,0]
    ax.imshow(total, alpha=alpha, cmap=cmap, vmin=1, vmax=7)
    ax.axis('off')
    fig.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    # Save plot
    plt.savefig(os.path.join(parameters.path_pm_predictions, 'C338C_MRI', cases[i], cases[i]+'_total_with_flair').replace("\\","/"))
    plt.close()

    # Plot prediction
    fig, ax = plt.subplots(figsize=(2,2))
    alpha = np.where(wmh == 0, 0, 1.0)[:,:,0]
    ax.imshow(wmh, alpha=alpha, cmap=cmap, vmin=1, vmax=7)
    ax.axis('off')
    fig.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    # Save plot
    plt.savefig(os.path.join(parameters.path_pm_predictions, 'C338C_MRI', cases[i], cases[i]+'_wmh').replace("\\","/"))
    plt.close()

    # Plot prediction
    fig, ax = plt.subplots(figsize=(2,2))
    alpha = np.where(nawm == 0, 0, 1.0)[:,:,0]
    ax.imshow(nawm, alpha=alpha, cmap=cmap, vmin=1, vmax=7)
    ax.axis('off')
    fig.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    # Save plot
    plt.savefig(os.path.join(parameters.path_pm_predictions, 'C338C_MRI', cases[i], cases[i]+'_nawm').replace("\\","/"))
    plt.close()

    # Plot prediction
    fig, ax = plt.subplots(figsize=(2,2))
    alpha = np.where(gm == 0, 0, 1.0)[:,:,0]
    ax.imshow(gm, alpha=alpha, cmap=cmap, vmin=1, vmax=7)
    ax.axis('off')
    fig.tight_layout()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
    plt.margins(0,0)
    # Save plot
    plt.savefig(os.path.join(parameters.path_pm_predictions, 'C338C_MRI', cases[i], cases[i]+'_gm').replace("\\","/"))
    plt.close()
    # plt.show()


def plot_pm_data(t1, fl, lfb2, lfb):
    # Plot t1, fl, lbl and prediction
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    ax[0].imshow(np.rot90(fl), cmap='gray')
    ax[0].set_title('T1')
    ax[0].axis('off')
    ax[1].imshow(np.rot90(fl), cmap='gray')
    ax[1].set_title('T1')
    ax[1].axis('off')

    ax[0].imshow(np.rot90(lfb), cmap='hot', alpha=0.5)
    ax[0].set_title('FLAIR')
    ax[0].axis('off')
    ax[1].imshow(np.rot90(lfb), cmap='hot', alpha=0.5)
    ax[1].set_title('FLAIR')
    ax[1].axis('off')

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