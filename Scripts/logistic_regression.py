#     This script can be used to test the performance of linear regression                         

import os, sys, ast
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import scipy.io as sio
from contextlib import redirect_stdout
from termcolor import colored

from helpers import *
from config import *
from split_reformat_data import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

################################################################################
#                            Functions                                         #
################################################################################

def grid_search_logistic(x_train, y_train, x_val, y_val, Cs =[1], degrees = [1]):
    '''Grid search using logistic regression of sklearn.
    
    Parameters
    ----------
    x_train, x_val : 2D numpy array
        Training and validation sets
    y_train, y_val : 1D numpy array
        Training and validation labels
    Cs : list of floats, optional
        Regularization constants to try (inverse, as in 
        sklearn.linear_model.LogisticRegression)
    degrees : list of integers, optional
        Polynomial augmentation degrees (with mixed terms) to try.
        
    Returns
    -------    
    resultsdf : pandas dataframe
        Dataframe with the results of the grid search
    '''
    resultsdf = pd.DataFrame(columns=['C','degree','train_acc','val_acc','model'])
    
    current_train = 1
    total_trs = len(Cs)*len(degrees)
    for degree in degrees:
        for C in Cs:
            print(f'Start training: {current_train}/{total_trs} ' +\
                                                f'(C = {C}, degree = {degree})')
            
            # Augment data
            poly = PolynomialFeatures(degree)
            tr_augmented = poly.fit_transform(x_train)
            val_augmented = poly.fit_transform(x_val)

            # Build and train the model
            logreg = linear_model.LogisticRegression(C=C)
            logreg = logreg.fit(tr_augmented, y_train)
            
            # Compute accuracies
            tr_acc = logreg.score(tr_augmented, y_train)
            val_acc = logreg.score(val_augmented, y_val)

            # Update df
            resultsdf.loc[current_train-1] = [C, degree, tr_acc, val_acc, 
                                                                       logreg]
            current_train = current_train +  1
            
    return resultsdf


def gen_data_for_logistic(data, labels, vid_size, slide):
    '''
    Generate the data to feed into a logistic regression model.

    The input data must have the shape of the 'raw data'. Meaning that the shape
    should be (n_awakenings, n_timesteps, n_features * 256).
    
    The ouptut array will be an array containing the flattened version of the 
    videos that we would have generated in the 'standard' pipeline. 
    The shape will be (n_awakenings * n_videos,  vid_size * n_features * 256).
    
    Parameters
    ----------
    data : numpy array
        Data to reformat. Shape: (n_awakenings, n_timesteps, n_features * 256)
    labels : numpy array
        Labels
    vid_size : integer
        The length of the videos that would have been generated in the standard
        pipeline. Here we do not generate videos but we simply take the 
        corresponding timesteps and we generate a 1D vector
    slide : integer
        The sliding window that would have been used in the standard pipeline.
        
    Returns
    -------    
    logistic_data : numpy array
        2D numpy array to be used in a linear/logistic model
    lostistic_labels : numpy array
        Same labels as before, but suitably repeated to match the length of the
        new data
    '''
    # Initialize empty lists
    logistic_data = []
    logistic_labels = []
    
    # Compute how many "videos" we have for each awakening
    num_video = (data.shape[1]-vid_size)//slide + 1
    
    # Generate the data from each awakening
    n_aw = data.shape[0]
    for i in range(n_aw):
        data_aw = data[i]
        lab_aw = labels[i]
        for j in range(num_video):
            logistic_data.append(data_aw[j*slide : j*slide+vid_size])
            logistic_labels.append(lab_aw)

    # Flatten the last part    
    logistic_data = np.array(logistic_data)
    logistic_data = logistic_data.reshape(n_aw * num_video, 
                                                       vid_size * data.shape[2])
    
    return logistic_data, np.array(logistic_labels)


def majority_voting_logistic(data, labels, degree, vid_size, slide, model):
    '''Majority voting applied to logistic case. 

    The input data must have the shape of the 'raw data'. Meaning that the shape
    should be (n_awakenings, n_timesteps, n_features * 256).
    
    Parameters
    ----------
    data : numpy array
        Data to reformat. Shape: (n_awakenings, n_timesteps, n_features * 256)
    labels : numpy array
        Labels
    degree : integer
        Polynomial degree expansion needed to use the model
    vid_size : integer
        The length of the videos that would have been generated in the standard
        pipeline. Here we do not generate videos but we simply take the 
        corresponding timesteps and we generate a 1D vector
    slide : integer
        The sliding window that would have been used in the standard pipeline.
        
    Returns
    -------    
    accuracy : float
        accuracy computed via mahority voting
    '''
    # number of awakenings in the dataset
    n_aw = len(data)
    acc1 = 0

    for aw in range(n_aw):
        # Generate logistic data for the single awakenings
        one_aw = np.reshape(data[aw], (1,data.shape[1],data.shape[2]))
        x, _ = gen_data_for_logistic(one_aw, labels, vid_size, slide)
        poly = PolynomialFeatures(degree)
        x_augmented = poly.fit_transform(x)

        # Use the model on the data coming from aw
        probabilities = model.predict_proba(x_augmented)
        
        # Sum the probabilities and then decise
        probabilities = probabilities.sum(axis=0)
        y_pred1 = (probabilities[1] > probabilities[0])*1
        
        # Update accuracy
        acc1 = acc1 + (y_pred1 == labels[aw])*1


    return acc1/n_aw
        
####################################################################################################
#                                          MAIN                                                    #
####################################################################################################

if __name__ == '__main__':
    
    # Select folder name, where the reults will be saved
    RESULTS_DIR = '../Results/' + 'Final/' + 'Logistic/'

    # Select the subjects to be used for the cross validation
    subject_IDs = ['H009', 'H018', 'H019', 'H021', 'H026', 'H025', 'H033', 'H035', 'H048', 'H050', 'H051', 'H054', 'H055', 'H057', 'H060', 'H061']

    # Decide if you want to subsample
    SUBSAMPLING_TRAIN = True
    SUBSAMPLING_VALIDATION = False

    # Choose the parameters to generate the "videos"
    VIDEO_SIZE = 10
    VIDEO_SLIDE = 5

    # Load data
    filename = MANIPULATED_DATA_DIR + 'FFT_data_SW_withFeatRatio_log_z-score.npz'
    print('Loading data from:', filename)
    datanpz = np.load(filename); data = datanpz['data']; 
    labels = datanpz['labels'];  labels = labels.flatten();

    # Load dataframe with infos
    df_filename = MANIPULATED_DATA_DIR + 'awakenings_info_df_FFT_SW.pkl'
    print('Loading infos dataframe from:', df_filename)
    infos_df = pd.read_pickle(df_filename)

    # Set index and add name_code column
    infos_df = infos_df.set_index(np.arange(len(infos_df)))
    infos_df['name_code'] = infos_df.apply(lambda row: row['Subject_id']+ '_' + row['Quest_number'], axis=1)
    
    # Check that the folder does not exists yet
    try:
        os.mkdir(RESULTS_DIR)
    except:
        print(colored(f'Directory {RESULTS_DIR} already exists!', 'red'))
        continue_bool = ask_user_0_1('Do you still want to continue? (0=NO/1=YES) ',
                                         print_0 = 'Exit.')
        if continue_bool == 0 :   
            exit()
            
    # Converting the labels and select the final dimension of the network
    print('----- Convert labels into DE vs NE ----- ')
    labels = classes3_to_classes2(labels)

    # Nb_channels and frames
    n_classes = len(np.unique(labels))
    print('Data shape:', data.shape)
    print('Number of classes:', n_classes)
    

    # Start leave out subject CV
    for subject_ID in subject_IDs:
        print(colored(f'\t\n----- Leaving out subject {subject_ID} -----\n','green'))

        # Find the codes of the awakenings belonging to the current subjectID
        temp = infos_df[infos_df['Subject_id'].str.contains(subject_ID)]
        test_codes = temp.name_code.values
        # Check that the subject code makes sense
        if len(test_codes) == 0:
            print(colored(f'{subject_ID}. ID not present in the dataset.', 'red'))
            print('Exit.')
            exit()

        # Split train and validation, validation made only of subject_ID
        print('Splitting train and validation. \nValidation awakenings:', test_codes)
        data_train, data_val, labels_train, labels_val = split_train_test(data, 
                    labels, infos_df, test_codes)
        
        # Subsampling the labels
        if SUBSAMPLING_TRAIN:
            print('Subsampling train:')
            data_train, labels_train = subsampling_labels(data_train, labels_train, shuffle = True, seed = 1)
        else:
            print('Labels in the training set:',np.bincount(labels_train.astype(int)))
        if SUBSAMPLING_VALIDATION:
            print('Subsampling validation:')
            data_val, labels_val = subsampling_labels(data_val, labels_val, shuffle = True, seed = 1)
        else:
            print('Labels in the validation set:', np.bincount(labels_val.astype(int)))

        # Split validation into test and training
        print('\nSplitting validation into validation (70%) and test (30%)...') 
        data_val, data_test, labels_val, labels_test = split_data(data_val, labels_val, 0.7, seed = 1)

        # Reformat data for logistic regression
        x_train, y_train = gen_data_for_logistic(data_train, labels_train, VIDEO_SIZE, VIDEO_SLIDE)
        x_val, y_val = gen_data_for_logistic(data_val, labels_val, VIDEO_SIZE, VIDEO_SLIDE)

        # Start the grid search        
        print('----- Calling grid-search train function -----')
        Cs = np.logspace(-7,3,20)
        degrees = [1] # not more than 1 (memory problems)
        resultsdf = grid_search_logistic(x_train, y_train, x_val, y_val, Cs, degrees)
        
        # Save a dataframe for each subject with grid search results                   
        df_filename = RESULTS_DIR + f'grid_search_results_{subject_ID}.pkl'
        resultsdf.to_pickle(df_filename)
        print(f'Dataframe saved in: {df_filename}' )

