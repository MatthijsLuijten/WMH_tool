import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import numpy as np
from tqdm import tqdm
import keras
from keras.preprocessing.image import ImageDataGenerator

from dataloader import load_data
from preprocess import preprocess
from model import *
from parameters import *
from plot import *
import utils


if __name__ == '__main__':

    # Either load and preprocess data OR load datasets if they are preprocessed already
    do_preprocess = False
    if do_preprocess:
        # Load data (case numbers) and shuffle
        print('--> Loading cases')
        train_cases, test_cases = load_data()

        # Make train and test datasets
        train_img = []#np.ndarray((len(train_cases)*111,200,200,2))
        train_lbl = []#np.ndarray((len(train_cases)*111,200,200,1))
        test_img = []#np.ndarray((len(test_cases)*111,200,200,2))
        test_lbl = []#np.ndarray((len(test_cases)*111,200,200,1))

        # Preprocess data and fill datasets
        print('--> Preprocessing training cases')
        for i,c in enumerate(tqdm(train_cases)):
            t1, fl, lbl = preprocess(c)
            for slice in range(len(t1[0][0])):
                if slice >= 39 and slice <= 149:
                    # train_img[slice + 111*i] = np.dstack((t1[:,:,slice],fl[:,:,slice]))
                    # train_lbl[slice + 111*i] = np.reshape(lbl[:,:,slice], (200,200,1))
                    train_img.append(np.dstack((t1[:,:,slice],fl[:,:,slice])))
                    train_lbl.append(np.reshape(lbl[:,:,slice], (200,200,1)))
        print('--> Preprocessing test cases')
        for i,c in enumerate(tqdm(test_cases)):
            t1, fl, lbl = preprocess(c)
            for slice in range(len(t1[0][0])):
                if slice > 39 and slice < 149:
                    # test_img[slice + 111*i] = np.stack((t1[:,:,slice],fl[:,:,slice]), axis=2)
                    # test_lbl[slice + 111*i] = np.reshape(lbl[:,:,slice], (200,200,1))
                    test_img.append(np.dstack((t1[:,:,slice],fl[:,:,slice])))
                    test_lbl.append(np.reshape(lbl[:,:,slice], (200,200,1)))

        # Array to np.array
        train_img = np.array(train_img) 
        train_lbl = np.array(train_lbl) 
        test_img = np.array(test_img) 
        test_lbl = np.array(test_lbl) 

        # Save datasets
        print('--> Saving datasets')
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

    # Make pred array for ensemble prediction
    model = build_unet(unet_input_shape)
    prediction = model.predict(np.array([train_img[0]]), batch_size=None)[0]
    prediction = np.ndarray(np.shape(prediction))
    
    for ensemble in range(training_ensemble):
        if train:
            # Prepare training data (generator)
            generator = ImageDataGenerator(validation_split=0.2)
            train_generator = generator.flow(train_img, train_lbl, batch_size=training_batch_size, subset='training', ignore_class_split=True)
            valid_generator = generator.flow(train_img, train_lbl, batch_size=training_batch_size, subset='validation', ignore_class_split=True)

            # Make model
            model = build_unet(unet_input_shape)

            # Train U-net
            print(f'--> Training model {ensemble+1}')
            # model_history = model.fit(train_img, train_lbl, epochs=training_epochs, validation_split=training_validation_split)
            model_history = model.fit(train_generator, validation_data=train_generator, epochs=training_epochs)
            
            # Plot training and save plot
            plot_training(model_history, ensemble)

            # Save model and parameters
            model.save(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/"))
            print('--> Saved model, training graph and parameters to', os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/"))
            with open('WMH_new/parameters.py', 'r') as f:
                txt = f.read()
                with open(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, "parameters.py").replace("\\","/"), 'w') as f:
                    f.write(txt)

        else:
            model = build_unet(unet_input_shape)
            model.load_weights(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/")).expect_partial()
            # model = keras.models.load_model(os.path.join(parameters.path_model_checkpoint, parameters.unet_version).replace("\\","/"), compile=False)#, custom_objects = {"iou_coef_loss": utils.iou_coef_loss, "coef_loss": utils.iou_coef})
            print(f'--> Loaded model {ensemble+1} from', os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/"))


        # Make prediction
        # for i in range(100,460,10):
        image_nr = 280
        pred_temp = model.predict(np.array([train_img[image_nr]]), batch_size=None)[0]
        prediction = prediction + pred_temp
    
    prediction = prediction / training_ensemble
    prediction[prediction[...,0] > 0.35] = 1
    prediction[prediction[...,0] <= 0.35] = 0
    metrics = {}
    metrics['IoU'] = round(utils.iou_coef(train_lbl[image_nr].astype('float64'), prediction).numpy(), 3)
    metrics['Dice'] = round(utils.dice_coef(train_lbl[image_nr].astype('float64'), prediction).numpy(), 3)
    plot_prediction(train_img[image_nr][:,:,0], train_img[image_nr][:,:,1], train_lbl[image_nr], prediction, metrics)
