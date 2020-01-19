


import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, Dropout, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow
get_ipython().run_line_magic('matplotlib', 'inline')

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
import os
import numpy as np
import tensorflow as tf
import keras




#from tensorflow.keras.preprocessing.image import ImageDataGenerator

#train_datagen = ImageDataGenerator(rescale = 1. / 255)
#train_generator = train_datagen.flow_from_directory(train_dir ,
                                                    #target_size = (224 , 224) ,
                                                    #class_mode = "categorical" )
#validation_generator = train_datagen.flow_from_directory(test_dir ,
                                                    #target_size = (224 , 224) ,
                                                    #class_mode = "categorical" )


batch_size = 32 
 
def inception_block(X, filters, stage, block):
    """  
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_path1 = X
    X_path2 = X
    X_path3 = X
    X_shortcut = X
    
    # First component of main path
    X_path1 = Conv2D(filters = F1, kernel_size = (1, 1), padding = 'same', name = conv_name_base + '2a')(X_path1)
    X_path1 = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X_path1)

    # Second component of main path
    X_path2 = Conv2D(filters = F2, kernel_size = (3, 3), padding = 'same', name = conv_name_base + '2b')(X_path2)
    X_path2 = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X_path2)

    # Third component of main path
    X_path3 = Conv2D(filters = F3, kernel_size = (5, 5), padding = 'same', name = conv_name_base + '2c')(X_path3)
    X_path3 = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X_path3)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = layers.Concatenate(axis = -1)([X_path1, X_path2, X_path3, X_shortcut])
    X = Activation('relu')(X)
    
    return X


def DeceptiNet(input_shape = (224,224,3), classes = 185):
     
    X_input = Input(input_shape)
#     X = ZeroPadding2D((16, 16))(X_input)

    X = Conv2D(8, (7, 7), padding = 'same', name = 'conv0', kernel_initializer = glorot_uniform(seed=0))(X_input)
    X = Conv2D(32, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((7, 7), strides=(2, 2))(X)
    
    X = inception_block(X, [32,32,32], 2, 'inception_1')
    
    X = Conv2D(256, (7, 7), strides = (2,2), name = 'conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv2')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    
    X = Conv2D(512, (7, 7), name = 'conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv3')(X)
    X = Activation('relu')(X)
    X = AveragePooling2D((3, 3), strides=(2, 2))(X)
    
    X = Flatten()(X)
    X = Dense(2048, activation='relu', name='fc-1')(X)
    X = Dense(1024, activation='relu', name='fc-2')(X)
    X = Dense(512, activation='relu', name='fc-3')(X)
    X = Dense(256, activation='relu', name='fc-4')(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    model = Model(inputs = X_input, outputs = X, name='DeceptiNet')
    return model


model = DeceptiNet()
model.summary()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_generator,
                              steps_per_epoch = 772,
                              epochs = 50,
                              validation_data = validation_generator,
                              validation_steps = 97,
                              verbose = 1)

from IPython.display import FileLink
FileLink(r'Final_Model.h5')





