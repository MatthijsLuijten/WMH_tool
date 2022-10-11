import matplotlib.pyplot as plt
import os
import parameters

def plot_training(model_history):
    # Plot the training and validation accuracy and loss at each epoch
    # print(model_history.history)
    # loss = model_history.history['loss']
    # val_loss = model_history.history['val_loss']
    acc = model_history.history['iou_coef']
    val_acc = model_history.history['val_iou_coef']

    plt.plot(model_history.epoch, acc, label='Training IoU Coeff.')
    plt.plot(model_history.epoch, val_acc, label='Validation IoU Coeff.')
    plt.title('Training and validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save plot
    if not os.path.exists(os.path.join(parameters.path_model_checkpoint, parameters.unet_version).replace("\\","/")):
            os.makedirs(os.path.join(parameters.path_model_checkpoint, parameters.unet_version).replace("\\","/"))
    plt.savefig(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, 'training_graph').replace("\\","/"))

    plt.show()


def plot_image(image, title):
    # Plot single slice of MRI data (ndarray)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()


def plot_prediction(t1_orig, fl_orig, label, prediction):
    # Plot t1, fl, lbl and prediction
    fig, ax = plt.subplots(1,4)
    ax[0].imshow(t1_orig, cmap='gray')
    ax[0].set_title('Original T1')
    ax[1].imshow(fl_orig, cmap='gray')
    ax[1].set_title('Original FL')
    ax[2].imshow(label, cmap='gray')
    ax[2].set_title('WMH label')
    ax[3].imshow(prediction, cmap='gray')
    ax[3].set_title('Prediction')
    plt.show()
