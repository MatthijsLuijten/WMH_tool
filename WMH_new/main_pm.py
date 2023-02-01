# Stop displaying warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE
from functools import partial

import numpy as np
from sklearn.utils import shuffle, class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from plot import *
from model import *
from multi_model import *
from parameters import *
from dataloader import load_pm_data
from preprocess import preprocess_pm_data
from utils import augment_data
import postprocess

if __name__ == '__main__':

    # Load and preprocess data 
    do_preprocess = False
    if do_preprocess:
        # Load data (case numbers) and shuffle
        print('--> Loading cases')
        train_cases, test_cases = load_pm_data(path_pm_wmh_cases)

        print('--> Preprocessing training and test cases')
        train_img, train_lbl_wmh, train_lbl_nawm, train_lbl_gm, test_img, test_lbl_wmh, test_lbl_nawm, test_lbl_gm = preprocess_pm_data(train_cases, test_cases)

        print('--> Saving datasets')
        save_pm_datasets(train_img, train_lbl_wmh, train_lbl_nawm, train_lbl_gm, test_img, test_lbl_wmh, test_lbl_nawm, test_lbl_gm)     


    # Make and fit model, OR load trained model
    train = False

    # Load test datasets
    print(f'--> Loading test datasets and making empty prediction arrays')

    # Make pred arrays for ensemble prediction
    test_img = np.load(path_pm_test_img)
    wmh_test_lbl = np.load(path_wmh_test_lbl)
    nawm_test_lbl = np.load(path_nawm_test_lbl)
    gm_test_lbl = np.load(path_gm_test_lbl)
    test_lbl = 3*wmh_test_lbl + 2*np.where(np.array(nawm_test_lbl-10*wmh_test_lbl) > 0.1, 1., 0.) + np.where(np.array(gm_test_lbl-10*nawm_test_lbl-10*wmh_test_lbl) > 0.1, 1., 0.)
    if 'multi' in parameters.unet_version:
        test_lbl_cat = to_categorical(test_lbl, 4)
        preds = [np.zeros((200,200,4)) for _ in range(len(test_img))]
    else:
        wmh_preds = [np.zeros((200,200,1)) for _ in range(len(test_img))]
        nawm_preds = [np.zeros((200,200,1)) for _ in range(len(test_img))]
        gm_preds = [np.zeros((200,200,1)) for _ in range(len(test_img))]
    
    for ensemble in range(training_ensemble):
        if train:
            if 'multi' in parameters.unet_version:
                # Load train datasets
                print(f'--> Loading and shuffling datasets for training model {ensemble}')
                train_img, train_wmh_lbl, train_nawm_lbl, train_gm_label = shuffle(np.load(path_pm_train_img), np.load(path_wmh_train_lbl), np.load(path_nawm_train_lbl), np.load(path_gm_train_lbl))
                train_lbl = 3*train_wmh_lbl + 2*np.where(np.array(train_nawm_lbl-10*train_wmh_lbl) > 0.1, 1., 0.) + np.where(np.array(train_gm_label-10*train_nawm_lbl-10*train_wmh_lbl) > 0.1, 1., 0.)

                # Augment data
                print(f'--> Augment datasets for training model')
                aug_train_img, aug_train_lbl = augment_data(train_img, train_lbl)
                # for i, t in enumerate(train_img):
                #     plot_augmented_data(train_img[i][:,:,0], train_img[i][:,:,1], train_lbl[i], aug_train_img[i][:,:,0], aug_train_img[i][:,:,1], aug_train_lbl[i])

                aug_train_lbl_cat = to_categorical(aug_train_lbl, 4)
                
                labels = np.ravel(aug_train_lbl)
                class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
                class_weights_dict = dict(zip(np.unique(labels), class_weights))
                print('Class weights are: ', class_weights)

                # Prepare training data (generator)
                gen = ImageDataGenerator(validation_split=0.2)
                train_generator = gen.flow(aug_train_img, aug_train_lbl_cat, batch_size=training_batch_size, subset='training', ignore_class_split=True)
                valid_generator = gen.flow(aug_train_img, aug_train_lbl_cat, batch_size=training_batch_size, subset='validation', ignore_class_split=True)

                # Make model
                model = build_multi_unet(unet_input_shape)

                # Create callbacks
                checkpoint_filepath = os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/")
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(checkpoint_filepath, "./logs"))
                model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_dice_coef_multilabel', mode='max', save_best_only=True, verbose=1)

            else:
                # Load train datasets
                print(f'--> Loading and shuffling datasets for training model {ensemble}')
                train_img, train_lbl = shuffle(np.load(path_pm_train_img), np.load(path_wmh_train_lbl))
                
                # Augment data
                print(f'--> Augment datasets for training model')
                aug_train_img, aug_train_lbl = augment_data(train_img, train_lbl)
                # for i, t in enumerate(train_img):
                #     plot_augmented_data(train_img[i][:,:,0], train_img[i][:,:,1], train_lbl[i], aug_train_img[i][:,:,0], aug_train_img[i][:,:,1], aug_train_lbl[i])

                # Prepare training data (generator)
                gen = ImageDataGenerator(validation_split=0.2)
                train_generator = gen.flow(aug_train_img, aug_train_lbl, batch_size=training_batch_size, subset='training', ignore_class_split=True)
                valid_generator = gen.flow(aug_train_img, aug_train_lbl, batch_size=training_batch_size, subset='validation', ignore_class_split=True)

                # Make model
                model = build_unet(unet_input_shape)

                # Create callbacks
                checkpoint_filepath = os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/")
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(checkpoint_filepath, "./logs"))
                model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_dice_coef', mode='max', save_best_only=True, verbose=1)


            # Train U-net
            print(f'--> Training model {ensemble}')
            model_history = model.fit(train_generator, validation_data=valid_generator, epochs=training_epochs, callbacks=[tensorboard_callback, model_checkpoint])
            
            # Plot training and save plot
            plot_training(model_history, ensemble)

            # Save model and parameters
            print('--> Saved model, training graph and parameters to', os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/"))
            with open('WMH_new/parameters.py', 'r') as f:
                txt = f.read()
                with open(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, "parameters.py").replace("\\","/"), 'w') as f:
                    f.write(txt)

        else: # Inference
            if 'multi' in parameters.unet_version:
                model = build_multi_unet(unet_input_shape)
                model.load_weights(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/")).expect_partial()
                print(f'--> Loaded model {ensemble+1} from', os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble)).replace("\\","/"))
                print('--> Make predictions')
                predictions = model.predict(test_img, batch_size=16)
                for i, p in enumerate(preds):
                    preds[i] += predictions[i]
                
            else:
                print('--> Making models')
                models = [build_unet(unet_input_shape) for _ in range(3)]
                model_names = ['pm_wmh_albumentations_v2', 'pm_nawm_albumentations_v2', 'pm_gm_albumentations_v2']

                print('--> Make predictions')
                for model, name, pred in zip(models, model_names, [wmh_preds, nawm_preds, gm_preds]):
                    model.load_weights(os.path.join(parameters.path_model_checkpoint, name, str(ensemble)).replace("\\","/")).expect_partial()
                    predictions = model.predict(test_img, batch_size=16)
                    for i, p in enumerate(pred):
                        pred[i] += predictions[i]


    # Average and threshold prediction
    if 'multi' in parameters.unet_version:
        preds = tf.math.argmax(preds, axis=-1)
        preds = preds[..., tf.newaxis]
        # preds = np.round(preds)
    else:
        wmh_preds, nawm_preds, gm_preds = np.where(np.divide(wmh_preds, training_ensemble) > 0.35, 1., 0.), np.where(np.divide(nawm_preds, training_ensemble) > 0.35, 1., 0.), np.where(np.divide(gm_preds, training_ensemble) > 0.35, 1., 0.)
        preds = wmh_preds + 2*np.where(np.array(nawm_preds-10*wmh_preds) > 0.1, 1., 0.) + 3*np.where(np.array(gm_preds-10*nawm_preds-10*wmh_preds) > 0.1, 1., 0.)
    

    # Postprocess
    preds, wmh_mask, nawm_mask, gm_mask = postprocess.run(test_img, preds)

    for i in range(len(preds)):
        plot_prediction(test_img[i][:,:,1], test_lbl[i], wmh_mask[i], nawm_mask[i], gm_mask[i], i)
        # plot_lbl_and_pred(test_img[i][:,:,1], test_img[i][:,:,1],  test_lbl[i], preds[i], i)

    # Evaluate metrics
    # utils.evaluate_metrics(preds, test_lbl)