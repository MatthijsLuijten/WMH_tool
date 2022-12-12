# Stop displaying warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import datetime
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import utils
from plot import *
from model import *
from parameters import *
from dataloader import load_data
from preprocess import preprocess

if __name__ == '__main__':

    # Either load and preprocess data OR load datasets if they are preprocessed already
    do_preprocess = False
    if do_preprocess:
        # Load data (case numbers) and shuffle
        print('--> Loading cases')
        train_cases, test_cases = load_data()
        # cases = load_data()

        # Make train and test datasets
        train_img, train_lbl, test_img, test_lbl = [], [], [], []

        # Preprocess data and fill datasets
        print('--> Preprocessing training cases')
        for i,c in enumerate(tqdm(train_cases)):
            t1, fl, lbl = preprocess(c)
            # determine_training(fl[:,:,77], lbl[:,:,77], fl[:,:,82], lbl[:,:,82], fl[:,:,97], lbl[:,:,97], fl[:,:,127], lbl[:,:,127], c)
            for slice in range(len(t1[0][0])):
                if slice >= 39 and slice <= 149:
                    train_img.append(np.dstack((t1[:,:,slice],fl[:,:,slice])))
                    train_lbl.append(np.reshape(lbl[:,:,slice], (200,200,1)))
                    
        print('--> Preprocessing test cases')
        for i,c in enumerate(tqdm(test_cases)):
            t1, fl, lbl = preprocess(c)
            for slice in range(len(t1[0][0])):
                if slice > 39 and slice < 149:
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
    

    # Make and fit model, OR load trained model
    train = True
    
    if not train:
        # Load test datasets
        print(f'--> Loading test datasets')
        # test_img, test_lbl = shuffle(np.load(path_test_img), np.load(path_test_lbl))
        test_img = np.load(path_test_img)
        test_lbl = np.load(path_test_lbl)

        # Make pred arrays for ensemble prediction
        model = build_unet(unet_input_shape)
        temp_pred = model.predict(np.array([test_img[0]]), batch_size=None)[0]
        temp_pred = np.ndarray(np.shape(temp_pred))
        preds = []
        for i in range(np.shape(test_img)[0]):
            preds.append(temp_pred)
    
    for ensemble in range(training_ensemble):
        if train:
            # Load train datasets
            print(f'--> Loading and shuffling datasets for training model {ensemble}')
            train_img, train_lbl = shuffle(np.load(path_train_img), np.load(path_train_lbl))
            
            # Prepare training data (generator)
            train_gen = ImageDataGenerator(validation_split=0.2, rotation_range=15, shear_range=18., zoom_range=0.1)
            valid_gen = ImageDataGenerator(validation_split=0.2)
            train_generator = train_gen.flow(train_img, train_lbl, batch_size=training_batch_size, subset='training', ignore_class_split=True)
            valid_generator = valid_gen.flow(train_img, train_lbl, batch_size=training_batch_size, subset='validation', ignore_class_split=True)

            # Make model
            model = build_unet(unet_input_shape)

            # Create callbacks
            checkpoint_filepath = os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(checkpoint_filepath, "./logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
            model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_dice_coef', mode='max', save_best_only=True, verbose=1)

            # Train U-net
            print(f'--> Training model {ensemble}')
            model_history = model.fit(train_generator, validation_data=train_generator, epochs=training_epochs, callbacks=[tensorboard_callback, model_checkpoint])
            
            # Plot training and save plot
            plot_training(model_history, ensemble)

            # Save model and parameters
            # model.save(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/"))
            print('--> Saved model, training graph and parameters to', os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/"))
            with open('WMH_new/parameters.py', 'r') as f:
                txt = f.read()
                with open(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, "parameters.py").replace("\\","/"), 'w') as f:
                    f.write(txt)

        else:
            model = build_unet(unet_input_shape)
            model.load_weights(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/")).expect_partial()
            print(f'--> Loaded model {ensemble+1} from', os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/"))


        # Make prediction
        if not train:
            # For 1 prediction
            image_nr = 150
            prediction = model.predict(np.array([test_img[image_nr]]), batch_size=None)[0]
            preds[0] = preds[0] + prediction

            # For full test set
            # print('   --> Make predictions')
            # predictions = model.predict(test_img, batch_size=16)
            # for i, p in enumerate(preds):
            #     preds[i] = p + predictions[i]
    
    # Average and threshold prediction
    preds = np.divide(preds, training_ensemble)
    preds = np.where(preds > 0.35, 1., 0.)
    plot_prediction(test_img[image_nr][:,:,0], test_img[image_nr][:,:,1], test_lbl[image_nr], preds[0])
    
    metrics = {}
    count = 0
    for i, p in enumerate(tqdm(preds)):
        # if np.sum(p) == 0 and np.sum(test_lbl[i]) == 0:
        if np.sum(test_lbl[i]) == 0:
            continue
        pred_nii = sitk.GetImageFromArray(p)
        label_nii = sitk.GetImageFromArray(test_lbl[i].astype('float64'))
        testImage, resultImage = getImages(label_nii, pred_nii)
        if not metrics:
            metrics['DSC'] = round(utils.getDSC(label_nii, pred_nii), 3)
            # metrics['HD'] = round(utils.getHausdorff(testImage, resultImage), 3)
            # metrics['AVD'] = round(utils.getAVD(testImage, resultImage), 3)
            metrics['AVD'] = round(utils.getAVD2(testImage, resultImage), 3)
            recall, f1 = utils.getLesionDetection(testImage, resultImage)
            metrics['Recall'] = round(recall, 3)
            metrics['F1'] = round(f1, 3)
        else:
            metrics['DSC'] = metrics['DSC'] + round(utils.getDSC(label_nii, pred_nii), 3)
            # metrics['HD'] = metrics['HD'] + round(utils.getHausdorff(testImage, resultImage), 3)
            # metrics['AVD'] = metrics['AVD'] + round(utils.getAVD(testImage, resultImage), 3)
            metrics['AVD'] = metrics['AVD'] + round(utils.getAVD2(testImage, resultImage), 3)
            recall, f1 = utils.getLesionDetection(testImage, resultImage)
            metrics['Recall'] = metrics['Recall'] + round(recall, 3)
            metrics['F1'] = metrics['F1'] + round(f1, 3)
        count = count + 1

    metrics = {k: v / count for k, v in metrics.items()}
    print(metrics)