import matplotlib.pyplot as plt

def plot_training(model_history):
    #plot the training and validation accuracy and loss at each epoch
    loss = model_history.history['loss']
    # val_loss = model_history.history['val_loss']
    acc = model_history.history['accuracy']

    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    # plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    plt.plot(model_history.epoch, acc, 'y', label='Training acc')
    plt.title('Training Loss and Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()


def plot_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.show()
