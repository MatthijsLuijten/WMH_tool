import numpy as np
from tqdm import tqdm

from dataloader import load_data
from preprocess import preprocess
from model import *
from plot import *


if __name__ == '__main__':

    # Load data
    train_cases, test_cases = load_data()
    # cases = train_cases + test_cases

    train_img = np.ndarray((len(train_cases),256,256,2))
    train_lbl = np.ndarray((len(train_cases),256,256,1))
    test_img = np.ndarray((len(test_cases),256,256,2))
    test_lbl = np.ndarray((len(test_cases),256,256,1))

    # # Preprocess data
    # print('--> Preprocessing training cases')
    # for i,c in enumerate(tqdm(train_cases)):
    #     t1, fl, lbl = preprocess(c)
    #     train_img[i] = np.stack((t1,fl), axis=2)
    #     train_lbl[i] = np.reshape(lbl, (256,256,1))
    # print('--> Preprocessing test cases')    
    # for i,c in enumerate(tqdm(test_cases)):
    #     t1, fl, lbl = preprocess(c)
    #     test_img[i] = np.stack((t1,fl), axis=2)
    #     test_lbl[i] = np.reshape(lbl, (256,256,1))

    # Load model
    input_shape = (256,256,2)
    model = build_unet(input_shape)
    pred_mask = model.predict(np.array([train_img[0]]), batch_size=None)
    plot_image(pred_mask[0], 'prediciton before training')

    # # Train U-net
    model_history = model.fit(train_img, train_lbl, batch_size=32, epochs=5, validation_split=0.2)

    # Plot training
    plot_training(model_history)

    # Make prediction
    pred_mask = model.predict(np.array([train_img[0]]), batch_size=None)
    plot_image(pred_mask[0], 'prediciton after training')