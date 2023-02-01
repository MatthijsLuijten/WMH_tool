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

from plot import *
from model import *
from parameters import *
from dataloader import load_data
from preprocess import preprocess_data

if __name__ == '__main__':

    # Load and preprocess data 
    do_preprocess = False
    if do_preprocess:
        # Load data (case numbers) and shuffle
        print('--> Loading cases')
        train_cases, test_cases = load_data()

        print('--> Preprocessing training and test cases')
        train_img, train_lbl, test_img, test_lbl = preprocess_data(train_cases, test_cases)
        
        print('--> Saving datasets')
        save_datasets(train_img, train_lbl, test_img, test_lbl)     


    # Make and fit model, OR load trained model
    train = False
    
    if not train:
        # Load test datasets
        print(f'--> Loading test datasets')
        test_img = np.load(path_test_img)
        test_lbl = np.load(path_test_lbl)

        # Make pred arrays for ensemble prediction
        preds = [np.zeros((182,218,1)) for _ in range(len(test_img))]
    
    for ensemble in range(training_ensemble):
        if train:
            # Load train datasets
            print(f'--> Loading and shuffling datasets for training model {ensemble+2}')
            train_img, train_lbl = shuffle(np.load(path_train_img), np.load(path_train_lbl))
            
            # Prepare training data (generator)
            train_gen = ImageDataGenerator(validation_split=0.2)
            valid_gen = ImageDataGenerator(validation_split=0.2)
            train_generator = train_gen.flow(train_img, train_lbl, batch_size=training_batch_size, subset='training', ignore_class_split=True)
            valid_generator = valid_gen.flow(train_img, train_lbl, batch_size=training_batch_size, subset='validation', ignore_class_split=True)

            # Make model
            model = build_unet(unet_input_shape)

            # Create callbacks
            checkpoint_filepath = os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble+2)).replace("\\","/")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(checkpoint_filepath, "./logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
            model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_dice_coef', mode='max', save_best_only=True, verbose=1)

            # Train U-net
            print(f'--> Training model {ensemble+2}')
            model_history = model.fit(train_generator, validation_data=valid_generator, epochs=training_epochs, callbacks=[tensorboard_callback, model_checkpoint])
            
            # Plot training and save plot
            plot_training(model_history, ensemble+2)

            # Save model and parameters
            model.save(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble+2)).replace("\\","/"))
            print('--> Saved model, training graph and parameters to', os.path.join(parameters.path_model_checkpoint, parameters.unet_version, str(ensemble+2)).replace("\\","/"))
            with open('WMH_new/parameters.py', 'r') as f:
                txt = f.read()
                with open(os.path.join(parameters.path_model_checkpoint, parameters.unet_version, "parameters.py").replace("\\","/"), 'w') as f:
                    f.write(txt)

        else:
            # Load a pre-trained model
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
            #     preds[i] += predictions[i]
    
    # Average and threshold prediction
    preds = np.divide(preds, training_ensemble)
    preds = np.where(preds > 0.35, 1., 0.)
    plot_prediction(test_img[image_nr][:,:,0], test_img[image_nr][:,:,1], test_lbl[image_nr], preds[0])
    

    # Evaluate metrics
    # utils.evaluate_metrics(preds, test_lbl)
