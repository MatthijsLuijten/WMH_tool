from dataloader import load_data
from preprocess import preprocess
from model import *


if __name__ == '__main__':

    # Load data
    train_cases, valid_cases, test_cases = load_data()
    cases = train_cases + valid_cases #+ test_cases

    # Preprocess data
    for c in cases:
        preprocess(c)

    # Load model
    input_shape = (256,256,2)
    model = build_unet(input_shape)