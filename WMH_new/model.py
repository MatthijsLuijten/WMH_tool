from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Dropout, Cropping2D, ZeroPadding2D
from keras.optimizers import Adam
from keras.layers import MaxPool2D, Concatenate

import parameters
from utils import *

#################### The U-Net Model ####################
#Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64, kernel_size=5)
    s2, p2 = encoder_block(p1, 96)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    b1 = conv_block(p4, 512) #Bridge

    d1 = decoder_block(b1, s4, 256)
    d2 = decoder_block(d1, s3, 128)
    d3 = decoder_block(d2, s2, 96)
    d4 = decoder_block(d3, s1, 64)

    ch, cw = get_crop_shape(inputs, d4)
    outputs = ZeroPadding2D(padding=(ch, cw))(d4)
    outputs = Conv2D(1, 1, activation="sigmoid")(outputs)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")

    model.compile(optimizer=Adam(learning_rate=parameters.unet_lr), loss = dice_coef_loss, metrics = dice_coef)
    # model.summary(150)

    return model

#################### The U-Net Model ####################


#################### Building blocks ####################
def conv_block(input, num_filters, kernel_size=3):
    x = Conv2D(num_filters, kernel_size, padding="same", activation='relu', kernel_initializer = 'he_normal')(input)
    x = Dropout(parameters.unet_dropout)(x)
    x = BatchNormalization()(x)   #Not in the original network. 
    # x = Activation("relu")(x)

    x = Conv2D(num_filters, kernel_size, padding="same", activation='relu', kernel_initializer = 'he_normal')(x)
    x = Dropout(parameters.unet_dropout)(x)
    x = BatchNormalization()(x)  #Not in the original network
    # x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling

def encoder_block(input, num_filters, kernel_size=3):
    x = conv_block(input, num_filters, kernel_size)
    p = MaxPool2D((2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = UpSampling2D((2, 2))(input)
    ch, cw = get_crop_shape(skip_features, x)
    crop_conv4 = Cropping2D(cropping=(ch,cw))(skip_features)
    x = Concatenate()([x, crop_conv4])
    x = conv_block(x, num_filters)
    return x

#################### Building blocks ####################
