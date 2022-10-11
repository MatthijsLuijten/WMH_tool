import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from tqdm import tqdm
import keras

from dataloader import load_data
from preprocess import preprocess
from model import *
from parameters import *
from plot import *
import utils


if __name__ == '__main__':

    # Either load and preprocess data or load datasets if they are preprocessed already
    preprocess = False
    if (preprocess):
        # Load data (case numbers)
        print('--> Loading and preprocessing cases')
        train_cases, test_cases = load_data()

        # Make train and test datasets
        train_img = np.ndarray((len(train_cases),256,256,2))
        train_lbl = np.ndarray((len(train_cases),256,256,1))
        test_img = np.ndarray((len(test_cases),256,256,2))
        test_lbl = np.ndarray((len(test_cases),256,256,1))

        # Preprocess data and fill datasets
        print('--> Preprocessing training cases')
        for i,c in enumerate(tqdm(train_cases)):
            t1, fl, lbl = preprocess(c)
            train_img[i] = np.dstack((t1,fl))
            train_lbl[i] = np.reshape(lbl, (256,256,1))
        print('--> Preprocessing test cases')    
        for i,c in enumerate(tqdm(test_cases)):
            t1, fl, lbl = preprocess(c)
            test_img[i] = np.stack((t1,fl), axis=2)
            test_lbl[i] = np.reshape(lbl, (256,256,1))

        # Save datasets
        np.save(path_train_img, train_img)
        np.save(path_train_lbl, train_lbl)
        np.save(path_test_img, test_img)
        np.save(path_test_lbl, test_lbl)

    else:
        # Load datasets
        print('--> Loading datasets')
        train_img = np.load(path_train_img)
        train_lbl = np.load(path_train_lbl)
        test_img = np.load(path_test_img)
        test_lbl = np.load(path_test_lbl)
        
    # Make and fit model, OR load trained model
    train = False
    if train:
        # Make model
        model = build_unet(unet_input_shape)

        # Train U-net
        print('--> Training model')
        model_history = model.fit(train_img, train_lbl, batch_size=training_batch_size, epochs=training_epochs, validation_split=training_validation_split)
        
        # Plot training and save plot
        plot_training(model_history)

        # Save model
        model.save(os.path.join(parameters.path_model_checkpoint, parameters.unet_version).replace("\\","/"))
        print('--> Saved model and training graph to', os.path.join(parameters.path_model_checkpoint, parameters.unet_version).replace("\\","/"))

    else:
        model = keras.models.load_model(os.path.join(parameters.path_model_checkpoint, parameters.unet_version).replace("\\","/"), custom_objects = {"iou_coef_loss": utils.iou_coef_loss, "coef_loss": utils.iou_coef})
        print('--> Loaded model from', os.path.join(parameters.path_model_checkpoint, parameters.unet_version).replace("\\","/"))


    # Make prediction
    pred_mask = model.predict(np.array([train_img[25]]), batch_size=None)
    plot_prediction(train_img[25][:,:,0], train_img[25][:,:,1], train_lbl[25], pred_mask[0])