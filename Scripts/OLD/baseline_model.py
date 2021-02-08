################################################################################
# Here there is the model definition and the functions to train the model      #
# identified by Ezgi as the best one.                                          #
#                                                                              #
# The code is basically the one of Ezgi, but here I kept only the part that is #
# needed by the """final""" model and I enlarged the grid_search               #
################################################################################

import os, sys, ast
import math as m
import numpy as np
import pickle


import pandas as pd
import scipy.io as sio

import tensorflow as tf
from tensorflow.python.keras.utils import np_utils, generic_utils
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import optimizers
from tensorflow.python.keras import initializers
from tensorflow.python.keras.layers import Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D, LSTM
from tensorflow.python.keras.layers.core import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import backend as K
from tensorflow.random import set_random_seed as set_seed_tensorflow
from sklearn.model_selection import KFold
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard

from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model



from from dreamUtils import *
from from dreamNetworks import *
from config import * 
from split_reformat_data import *


def CNN_Video_Multi(nb_classes=3, dropoutRate = 0.5, act='relu', k_size=3, 
                    d_layer = 512, L2_regularizer = 0.001, 
                    img_size=32, time_slot = 10, num_color_chan=2):
    """ 
    Deep convolutional 3D neural network with softmax classifier
    
    :param nb_classes: number of class
    :param dropoutRate: drop-out rate of last layer
    :param act: activation function
    :param k_size: convolutional kernel size
    :param k_regularizer: kernel regularizer
    :param d_layer: number of hidden unit in the last layer
    :param img_size: image size
    :param time_slot: number of frames/images in a video, length of the video
    :param num_color_chan = number of color channel in the image/frame, no RGB 
        values used but delta and beta-gamma power
    values of electrodes are used
    Expecting 100x32x32x1 video data as input
      
    Conv3D<32> - Conv3D<32> - Conv3D<32> - Conv3D<32> - MaxPool3D<2,2,2> - 
    Conv3D<64> - Conv3D<64> - MaxPool3D<2,2,2> - 
    Dense<512> - Dense<3>
     
    """
    strides = None
    kernel = (2, k_size, k_size)
    k_regularizer = regularizers.l2(L2_regularizer)
    print(f'PARAMETERS OF MODELS: activation = {act}, kernel size = {k_size}, '+
        f'dropout = {dropoutRate}, regularizer = {L2_regularizer}, '+
        f'final_layer = {d_layer}')
     
    model = Sequential()
    # add layers
    model.add(Conv3D(32, kernel_size=kernel, 
              input_shape=(time_slot,img_size,img_size,num_color_chan), 
              activation=act, kernel_regularizer=k_regularizer))
    model.add(Conv3D(32, kernel_size=kernel, padding='same',  activation=act,
              kernel_regularizer=k_regularizer))
    model.add(Conv3D(32, kernel_size=kernel, padding='same', 
              activation=act,
              kernel_regularizer=k_regularizer ))
    model.add(Conv3D(32, kernel_size=kernel, padding='same', 
              activation=act,
              kernel_regularizer=k_regularizer ))
    model.add(MaxPooling3D(pool_size=(2,2,1), strides=strides, 
              data_format='channels_last'))
    # new layer
    model.add(Conv3D(64, kernel_size=kernel, padding='same', 
              activation=act,
              kernel_regularizer=k_regularizer))
    model.add(Conv3D(64, kernel_size=kernel, padding='same', 
              activation=act,
              kernel_regularizer=k_regularizer))
    model.add(MaxPooling3D(pool_size=(2,2,2),strides=strides, 
              data_format='channels_last'))
    
    # flatten and check
    model.add(Flatten())
    model.add(Dense(d_layer))
    model.add(Dropout(rate=dropoutRate))
    model.add(Dense(nb_classes, activation='softmax',
            kernel_regularizer=k_regularizer))
    
    return model
    
    
    
def grid_search(X_train, Y_train, X_valid, Y_valid, nb_classes, model_type, output_dir,
             patience = 10, batch_size=32, num_epochs=10):
    """
    Tune hyper-parameters of models via grid search
    :param sample: boolean, if true data is downsampled with fraction of 0.2 
    :param filename: data file name
    :param model_type: model to be tested
    :param batch_size: batch_size of SGD
    :param num_epochs:  number of epochs of training 
    :param run_id: runner identifier 
    """
    ############################ Play with parameters ##########################
    dropoutRates = [0.30,0.50,0.70]
    acts= ['relu']
    k_sizes= [3,5]
    d_layers = [128,256]
    l2_regularizers = [0.001, 0.0001] #l2 regularizer

    ############################################################################

    # Help usability: save results in dataframe and print progress
    current_train = 1
    total_parameters = len(dropoutRates)*len(acts)*len(d_layers)*\
                        len(k_sizes)*len(l2_regularizers)
    resultsdf = pd.DataFrame(columns=['Accuracy', 'Description'])

    # Early stopping
    es_val_loss = EarlyStopping(monitor='val_loss', mode='min', patience = patience, verbose=1)
   
    if model_type == 'Video-Multi':

        for act in acts:
            for k_size in k_sizes:
                for d_layer in d_layers:
                    for dropoutRate in dropoutRates:
                        for l2_regularizer in l2_regularizers:
                            
                            print('\n--------------------------------------')
                            print('\tSTART TRAIN n. '+
                                 f'{current_train}/{total_parameters}')
                            print('---------------------------------------\n')

                            model = CNN_Video_Multi(nb_classes=nb_classes, 
                                                dropoutRate = dropoutRate, 
                                                act=act, k_size=k_size, 
                                                d_layer = d_layer, 
                                                L2_regularizer = l2_regularizer)

                            print('\n\n')
                            sgd =  optimizers.SGD(lr=0.001, decay=1e-6, 
                                momentum=0.9, 
                                nesterov=True) #this is for multichannel video

                            model.compile(optimizer='adam', 
                                loss='categorical_crossentropy', 
                                metrics=['accuracy'])
                            
                            SEED = 232323
                            set_seed_tensorflow(23)
                            np.random.seed(23)

                            # Tensorboard callback
                            tb_callback = TensorBoard(log_dir = output_dir + f'Graph_train{current_train}/',
                                        histogram_freq = 0, write_graph = True, write_images=True)
                            model.fit(X_train, Y_train, 
                                    validation_data=(X_valid, Y_valid), 
                                    epochs=num_epochs, 
                                    batch_size=batch_size, verbose=2,
                                    callbacks = [es_val_loss, tb_callback]
                                    )

                            # Calculate the metrics specified for training
                            scores = model.evaluate(X_valid, Y_valid)
                            print('Validation accuracy:', scores[1])

                            #Save result in a dataframe
                            description = f'activation = {act}, kernel size = '\
                               f'{k_size}, dropout = {dropoutRate}, regularizer'\
                               f' = {l2_regularizer}, final_layer = {d_layer}'
                            resultsdf.loc[current_train-1] = [scores[1], description]

                            # Counter update
                            current_train +=1


    else:
        raise ValueError("Model not supported []")

    return resultsdf


def train(X_train, Y_train, X_valid, Y_valid, model, 
            output_dir, batch_size=32, num_epochs=50, patience = 8):
    """
    Train models

    :param sample: boolean, if true data is downsampled with fraction of 0.2 
    :param filename: data file name
    :param model: model to be tested
    :param batch_size: batch_size of SGD
    :param num_epochs:  number of epochs of training 
    """

    np.random.seed(1234)
    set_seed_tensorflow(1234)

    # Compiling the model
    sgd =  optimizers.SGD(lr=0.0005, decay=1e-6, momentum=0.9, 
                                nesterov=True) #this is for multichannel video
    model.compile(optimizer=sgd, loss='categorical_crossentropy', 
                    metrics=['accuracy'])
    
    # Callbacks
    es_val_loss = EarlyStopping(monitor='val_loss', mode='min', patience = patience, verbose=1)
    tb_callback = TensorBoard(log_dir = output_dir + 'Graph/',
                    histogram_freq = 0, write_graph = True, write_images=True)   

    #model.summary()

    try:
        model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), 
                 epochs=num_epochs, batch_size=batch_size, verbose=2,
                 callbacks = [es_val_loss, tb_callback])
    except KeyboardInterrupt:
        print('\n\nKeyboard interruption!\n\n')
        pass

    prediction = model.predict_classes(X_valid)
    # Count number of unique target values for each possible one
    print(np.bincount(prediction))
    
    # Calculate the metrics specified for training
    scores = model.evaluate(X_valid, Y_valid)
    print('Validation accuracy:', scores[1])

    # Save the model
    weights_filename = output_dir +'weights.h5'
    print('Saving weights at', weights_filename)
    model.save(weights_filename)

    # return the accuracy
    return scores[1]



