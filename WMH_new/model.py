import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate

import parameters

#################### The U-Net Model ####################
#Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024) #Bridge

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")
    model.compile(optimizer=Adam(parameters.unet_lr), loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary(150)

    return model

#################### The U-Net Model ####################


#################### Building blocks ####################
def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same", activation='relu', kernel_initializer = 'he_normal')(input)
    x = Dropout(parameters.unet_dropout)(x)
    x = BatchNormalization()(x)   #Not in the original network. 
    # x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same", activation='relu', kernel_initializer = 'he_normal')(x)
    x = Dropout(parameters.unet_dropout)(x)
    x = BatchNormalization()(x)  #Not in the original network
    # x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters):
    x = UpSampling2D((2, 2))(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

#################### Building blocks ####################
