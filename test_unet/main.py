from blocks import build_unet
from keras.optimizers import Adam
from keras.utils import plot_model

input_shape = (256,256,2)

model = build_unet(input_shape)
model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Print graph
# dot_img_file = '/tmp/model_1.png'
# plot_model(model, to_file='graph.png', show_shapes=True)