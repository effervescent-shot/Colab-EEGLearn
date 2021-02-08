################################################################################
# Author: Nihal Ezgi Yuceturk, EPFL                                            #
# Source: https://github.com/effervescent-shot/Dream-Prediction                #
#                                                                              #
# Small changes by: Nicola Ischia                                              #
# Date:                                                                        #
################################################################################

#!/usr/bin/python

import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import pandas as pd
import scipy.io as sio

from tensorflow.python.keras.utils import np_utils

from config import DATA_DIR, MANIPULATED_DATA_DIR
from from dreamUtils import *
np.random.seed(19762326)


channels_coord = DATA_DIR + '/dataset_info/channelcoords.mat'
locs_3D = sio.loadmat(channels_coord)['locstemp']
locs_2D = map_to_2d(locs_3D)
print(locs_2D.shape)



def create_images(image_size, input_filename = "20sec_raw_data_zip",
                         output_filename = '_last20sec_img',
                         train_ratio = 0.9,
                         balance_train = True,
                         centerize = True):
    '''
    Images created from from provided dataset. Each image assigned the label of 
    the trial it comes from.

    :param image_size: Size of the images to created.
    :param filename: name of the file that has to be loaded
    :output filename: name of the file that will be saved
    '''
    # Load data
    filename = MANIPULATED_DATA_DIR + input_filename
    loaded = np.load(filename+'.npz')
    data = loaded['data']
    label = loaded['labels']
    
    print(f'Data loaded from {filename}.npz')
    print('Data shape:',data.shape)
    print('N. features for each electrode:', data.shape[2] // 256)
    
    # Centerize raw data for reference electrode
    if centerize:
        print('Subtracting the mean (is this correct with 2 features?)')
        data_normalized = []
        for trial in data:
            n_trial =  centerize_reference(trial)
            data_normalized.append(n_trial)
        data = np.array(data_normalized)   
    
    # Data is splitted train, validation and test sets  
    x_tr, y_tr, x_va, y_va, x_te, y_te = tt_split(data, label, train_ratio)

    # Labels (0 = NE, 1 = DEWR, 2 = DE)
    lab0 = np.squeeze(np.argwhere(y_tr == 0))
    lab1 = np.squeeze(np.argwhere(y_tr == 1))
    lab2 = np.squeeze(np.argwhere(y_tr == 2))
    np.random.shuffle(lab1)
    np.random.shuffle(lab2)    

    # Train balancing
    if balance_train:
        print('Balancing the dataset...')
        # Half of DEWR and half of DE are icluded. All NE included.
        temp = np.hstack( ( lab0, lab1[:int(len(lab1)*0.5)] ) )
        indices = np.hstack( ( temp, lab2[:int(len(lab2)*0.5)] ) )
    else:
        temp = np.hstack( ( lab0, lab1 ) )
        indices = np.hstack( ( temp, lab2 ) )

    # If balancing, this will balance the train set
    x_tr_new = x_tr[indices]
    y_tr_new = y_tr[indices]
    
    # Print informations about the train set
    unique, counts = np.unique(y_tr_new , return_counts=True)
    print('Label distribution in the train set:\n', np.asarray((unique, counts)).T)
    unique, counts = np.unique(y_va , return_counts=True)
    print('Label distribution in the validation set:\n', np.asarray((unique, counts)).T)
    unique, counts = np.unique(y_te , return_counts=True)
    print('Label distribution in the test set:\n', np.asarray((unique, counts)).T)
    
    #Create train matrix  
    X_train = np.concatenate(x_tr_new, axis = 0)
    y_train = np.concatenate([lab(y, x_tr_new[0].shape[0]) for y in y_tr_new]) 
    #Create validation matrix
    X_valid = np.concatenate(x_va, axis=0)
    y_valid = np.concatenate( [lab(y, x_va[0].shape[0]) for y in y_va]  )
    #Create test matrix
    X_test = np.concatenate(x_te, axis=0)
    y_test = np.concatenate( [lab(y, x_te[0].shape[0]) for y in y_te]  )
    
    print('Generating train images...')
    train_images = gen_images(locs_2D, X_train, image_size, normalize=True)
    print('Generating valid images...')
    valid_images = gen_images(locs_2D, X_valid, image_size, normalize = True)
    print('Generating test images...')
    test_images = gen_images(locs_2D, X_test, image_size, normalize=True)
      
    # Save results
    fileName =  MANIPULATED_DATA_DIR + \
                str(image_size)+'_'+str(image_size)+ output_filename
    np.savez_compressed(fileName, 
        train_img=train_images, train_labels=y_train, 
        test_img=test_images, test_labels=y_test, 
        valid_img=valid_images, valid_labels = y_valid )
    print(f'Saved in {fileName}.npz')
    print('(NPZ archive with keys: train_img, train_labels, test_img, test_labels, valid_img, valid_labels)')

def main():
    image_size = 32
    try:
        image_size = int(sys.argv[1])
    except Exception  as e:
        print('Please identify image size; default 32')
    print ('Start interpolate images of ',image_size, 'x', image_size, 'x', 1)    
    create_images_single(image_size)
    print ('Start interpolate images of ',image_size, 'x', image_size, 'x', 2)    
    create_images_multi(image_size)
    print('All done! Images are created and dumped.')  
        
if __name__ == '__main__':
      main()      
    
    
    
    
    
        
        
        
        
