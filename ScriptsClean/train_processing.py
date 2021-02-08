import os
import sys
import ast
import numpy as np
import pandas as pd
import scipy.io as sio
from termcolor import colored
from config import *


def split_train_test(data, labels, infos_df, test_codes, reformat = False):
    """
    This function is used to split the train and test from the original data. 
    The way to split train and test is that we find the awaking recoidings with most balanced labels as 
    the test set. The names of awaking recordings were saved in 'test_codes'.    
    If reformat is set to TRUE, axis 3 is moved to the final position.

    :param data: the original data.
    :param labels: the original labels.
    :param infos_df : the dataframe that contains each awaking recording information.
    :param test_codes : the name of the awakening recordings.
    :return: train data, test data, train labels, test labels
    """
    #reset the index of dataframe
    infos_df = infos_df.set_index(np.arange(len(infos_df)))
    #add the name_code column
    infos_df['name_code'] = infos_df.apply(lambda row: row['Subject_id']+ '_' + row['Quest_number'], axis=1)


    #find the test index
    test = infos_df[infos_df['name_code'].isin(test_codes)]
    test_index = test.index      
    #find the train index
    train = infos_df[~infos_df['name_code'].isin(test_codes)]
    train_index = train.index
    
    #check that the codes were correct
    if (len(test) != len(test_codes)) or (len(test) == 0 and len(test_codes) != 0):
        print(colored(f'Not able to find all the codes {test_codes}','red'))
        print('Exit.')
        sys.exit()
    
    #check that we don't loose data
    if (infos_df.index != (train_index.union(test_index))).any():
        print(colored(f'There are some data has been lost in the split data using codes: {test_codes}','red'))
        print('Exit.')
        sys.exit()        
    
    #split data to train and test accodirding to the index
    data_train = data[train_index]
    labels_train = labels[train_index]
    data_test = data[test_index]
    labels_test = labels[test_index]
        
    #reformat data 
    if reformat:
        print('Moving axis 3 to the last position.')
        print('Shape before the operation (train):', data_train.shape)
        print('Shape before the operation (test):', data_test.shape)
        data_train = np.moveaxis(data_train, 3, -1)
        data_test = np.moveaxis(data_test, 3, -1)
        print('Shape after the operation (train):', data_train.shape)
        print('Shape after the operation (test):', data_test.shape)

    return data_train, data_test, labels_train, labels_test    


def classes3_to_classes2(labels):
    """
    This function is used to change the 3 classes to 2 classes.
    
    Choose the labels: NE(1) and DE(2)
    label 0 means: NE
    label 1 means: DEWR
    label 2 means: DE
    
    we will change label 1 and label 2 together as label 1, we could name this new label 1 as DE_all.   
    """
    labels[labels == 2] = 1
    
    return labels    

def classes3_to_classes2_NE_DE(data, labels):
    """
    This function is used to extract the data of label0 and label2, and name label2 as 1.
    
    Choose the labels: NE(1) and DE(2)
    label 0 means: NE
    label 1 means: DEWR
    label 2 means: DE
    """

    lab0 = np.squeeze(np.argwhere(labels == 0))
    lab2 = np.squeeze(np.argwhere(labels == 2))
    indices_train = np.concatenate((lab0,lab2), axis=0)

    
    data = data[indices_train]
    labels = labels[indices_train] 
    labels[labels == 2] = 1
   
    return data, labels    



def split_data(x, y, ratio, seed=1, keep_proportions=1):
    """
    Split the dataset based on the split ratio. If ratio is 0.8 you will have 80% of your data 
    set dedicated to training and the rest dedicated to testing.
    
    Return the training then testing sets (x_tr, x_te) and training then testing labels (y_tr, y_te)

    :param x: the data
    :param y: the labels
    :param ratio: the ratio used to split data in two subsets
    :param seed: seed of RNG used to perform the split
    :param keep_proportions: keep the dream/no_dream distribution of the original dataset in the two subsets, 
                             by splitting separately each class. 
    """
    if keep_proportions == 0:
        #Set seed and permute
        np.random.seed(seed)
        xrand = np.random.permutation(x)
        np.random.seed(seed)
        yrand = np.random.permutation(y)
        #Used to compute how many samples correspond to the desired ratio.
        limit = int(y.shape[0]*ratio)
        x_tr = xrand[:limit]
        x_te = xrand[(limit):]
        y_tr = yrand[:limit]
        y_te = yrand[(limit):]
    else:
        #Find NO dreaming data
        NDE = np.where(y == 0)
        x_NDE = x[NDE]
        y_NDE = y[NDE]

        #Find dreaming data
        DE = np.where(y != 0)
        x_DE = x[DE]
        y_DE = y[DE]       

        #Set seed and permute
        np.random.seed(seed)
        x_NDErand = np.random.permutation(x_NDE)
        np.random.seed(seed)
        x_DErand = np.random.permutation(x_DE)    
        np.random.seed(seed)
        y_NDErand = np.random.permutation(y_NDE)
        np.random.seed(seed)
        y_DErand = np.random.permutation(y_DE)
    
            
        #Used to compute how many samples correspond to the desired ratio.
        limit_NDE = int(x_NDE.shape[0]*ratio)
        limit_DE = int(x_DE.shape[0]*ratio)
        
        x_NDE_tr = x_NDErand[:limit_NDE]
        x_NDE_te = x_NDErand[(limit_NDE):]
        x_DE_tr = x_DErand[:limit_DE]
        x_DE_te = x_DErand[(limit_DE):]
        x_tr = np.concatenate((x_NDE_tr,x_DE_tr),axis=0)
        x_te = np.concatenate((x_NDE_te,x_DE_te),axis=0)
        
        y_NDE_tr = y_NDErand[:limit_NDE]
        y_NDE_te = y_NDErand[(limit_NDE):]
        y_DE_tr = y_DErand[:limit_DE]
        y_DE_te = y_DErand[(limit_DE):]
        y_tr = np.concatenate((y_NDE_tr,y_DE_tr),axis=0)
        y_te = np.concatenate((y_NDE_te,y_DE_te),axis=0)
        

    return x_tr, x_te, y_tr, y_te


def subsampling_labels(data, labels, shuffle = True, seed = 1):
    """
    This function performs subsampling to get a dataset with perfectly balanced labels.
    Basically, we take the labels with the minumun count and reduce all the labels as the same count
    of this minumun count.
    Working both with 2 classes and 3 classes classification tasks.
    
    :param data: the data matrix
    :param labels : the labels column
    :param shuffle: shuffle data before subsampling
    :return: the subsampled data, the subsampled labels
    """
    if shuffle:
        np.random.seed(seed)
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        data = data[idx]
        labels = labels[idx]


    unique, counts = np.unique(labels.astype(int) , return_counts=True)    
    min_num = min(counts)
    
    Label_pos = []
    for i in unique:        
        label_pos = np.squeeze(np.argwhere(labels == i))
        label_pos = label_pos[0:min_num]
        Label_pos.append(label_pos)
    Label_pos = np.array(Label_pos).flatten()
    
    data = data[Label_pos]
    labels = labels[Label_pos]
    
    # Print label count of datasets by trial
    unique0, counts0 = np.unique(labels.astype(int) , return_counts=True)
    print('Label distribution:\n', np.asarray((unique0, counts0)).T)
    
    return data, labels


def reformat_data_labels(data, labels, move_axis = True):
    """
    This function will reformat data as awaking * n_videos ad also at the same time expands labels 
    according to the data.
    After this, all videos will be merged and not separated in awakenings anymore.
    
    :param data: the data matrix
    :param labels : the labels column
    :return: the reshaped data, the reshaped labels
    """

    # merge awakenings
    shp = data.shape
    data_reshaped = data.reshape(shp[0]*shp[1], shp[2], shp[3], shp[4], shp[5])
    # repeat labels
    n_rep = data.shape[1]
    labels_reshaped = np.repeat(labels, n_rep)

    # move axis
    if move_axis:    
        data_reshaped = np.moveaxis(data_reshaped, 2, -1)
    
    return data_reshaped, labels_reshaped
