from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate


def conv_block(input, num_filters, dropout_rate):
    x = Conv2D(num_filters, 3, padding="same", activation='relu')(input)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)   #Not in the original network. 
    # x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same", activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)  #Not in the original network
    # x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling

def encoder_block(input, num_filters, dropout_rate):
    x = conv_block(input, num_filters, dropout_rate)
    p = MaxPool2D((2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters, dropout_rate):
    x = UpSampling2D((2, 2))(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, dropout_rate)
    return x

#Build Unet using the blocks
def build_unet(input_shape):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64, 0.1)
    s2, p2 = encoder_block(p1, 128, 0.1)
    s3, p3 = encoder_block(p2, 256, 0.1)
    s4, p4 = encoder_block(p3, 512, 0.1)

    b1 = conv_block(p4, 1024, 0.1) #Bridge

    d1 = decoder_block(b1, s4, 512, 0.1)
    d2 = decoder_block(d1, s3, 256, 0.1)
    d3 = decoder_block(d2, s2, 128, 0.1)
    d4 = decoder_block(d3, s1, 64, 0.1)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)  #Binary (can be multiclass)

    model = Model(inputs, outputs, name="U-Net")
    return model