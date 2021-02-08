################################################################################
# Load already preprocessed data                                               #
# Divide it train/validation/test, prepare data for the training process       #
# Begin training process                                       #
#                                                                              #
# train/test/validation divided depending on subject                           #
################################################################################

import os, sys, ast
import numpy as np
import pandas as pd
import scipy.io as sio
from contextlib import redirect_stdout
from termcolor import colored

sys.path.append('..')
#from helpers import *
from config import *
from data_augmentation import *
from split_reformat_data import *
from feature_augmentation import *
from create_images_videos import *

# pytorch stuff
from pytorch_data_utils import *
from pytorch_models import *
from pytorch_train_utils import *




def define_traininig_parameters():
    # Select folder name, where the reults will be saved
    RESULTS_DIR = '../../Results/' + 'Final/weighted_loss_reduced_CNN_6_one_more_dp/' 
    
    # Select the filename of the images (without extension)
    images_filename = '../' + MANIPULATED_DATA_DIR + '32_32_FFT_withFeatRatio_log_z-score.npz'

    # Select video parameters
    VIDEO_SIZE = 10
    SLIDE = 3

    # Select the subjects to be used for the cross validation
    #subject_IDs = ['H009', 'H018', 'H019', 'H021', 'H026', 'H025', 'H033', 'H035', 
    #               'H048', 'H050', 'H051', 'H054', 'H055', 'H057', 'H060', 'H061']   
    subject_IDs = ['H019','H026','H048']
    
    # Which model?
    MODEL_TYPE = 'LSTM_reduced' #'mix' 2Dconv_max' 'convpool_conv1d' 'LSTM' 'CNN' 'FC'

    # Decide if you want to subsample
    SUBSAMPLING_TRAIN = True
    SUBSAMPLING_VALIDATION = False

    # Decide the weights for the loss ##### WHAT?
    w1 = 4
    w2 = 1
    LOSS_WEIGHTS = [w1/(w1+w2),w2/(w1+w2)] 

    # Decide how to treat labels. If True: 2 -> 1
    CONVERT_LABELS = True

    # Decide if you want to take only a portion of the videos
    ONLY_LAST_VIDEOS_TRAIN = False # Take last videos in training set
    ONLY_LAST_VIDEOS_VAL = False # Take last videos in validation set
    ratio_last_videos = 0.5         ##### WHAT? 

    # Decide if you want to add the sleep stage as a feature
    ADD_SLEEP_STAGE = True
    SLEEP_STAGE_METHOD = 'add' #multiply'  'add'

    # Decide if you want to use only NREM awakenings
    USE_ONLY_NREM = False

    # Decide if you want to do data augmentation
    AUGMENT_DATA = False
    # Parameters for data augmentation (if it is done)
    hor_tr = [1,-1]
    ver_tr = [0]
    noiseSTD = 0

    # Check that the folder where the results are saved does not exists yet
    try:
        os.mkdir(RESULTS_DIR)
    except:
        print(colored(f'Directory {RESULTS_DIR} already exists!', 'red'))
        continue_bool = ask_user_0_1('Do you still want to continue? (0=NO/1=YES) ', print_0 = 'Exit.')
        if continue_bool == 0 :   
            exit()
            
            


if __name__ == '__main__':

    # Select the GPU (at runtime)
    GPU = ask_user_integer('Which GPU do you want to use? (type -1 for cpu) ')
    
    # Define necessary train params above, not good way to do it but this people ...
    define_traininig_parameters()

    # Generate videos
    data, labels = create_video(images_filename, VIDEO_SIZE, SLIDE)
    labels = labels.flatten()

    # Load dataframe with infos
    df_filename = '../' + MANIPULATED_DATA_DIR + 'awakenings_info_df_FFT_SW.pkl'
    print('Loading infos dataframe from:', df_filename)
    infos_df = pd.read_pickle(df_filename)

    # Set index and add name_code column
    infos_df = infos_df.set_index(np.arange(len(infos_df)))
    infos_df['name_code'] = infos_df.apply(lambda row: row['Subject_id']+ '_' + row['Quest_number'], axis=1)

    # Converting the labels and select the final dimension of the network
    if CONVERT_LABELS:
        print('----- Convert labels into DE vs NE ----- ')
        labels = classes3_to_classes2(labels)

    # Add ratio of 2 features (do this before adding sleep stage!)
    # NOTE: this should not be necessary, the ratio should be already
    # present in the video!
    # Decide if you want to add the ratio of two band as feature
    # ADD_RATIO = False
    # if ADD_RATIO:
    #     print(f'----- Adding RATIO of features -----')
    #     channel_num = 0; channel_den = 1;
    #     data = add_feature_ratio_to_all_videos(data, channel_num, channel_den)

    # Add sleep stage as a feature
    if ADD_SLEEP_STAGE:
        print(f'----- Adding REM/NREM feature with method: {SLEEP_STAGE_METHOD} -----')
        data = add_sleep_stage_to_all_videos(data, infos_df, method = SLEEP_STAGE_METHOD)

    # Use only NREM
    if USE_ONLY_NREM:
        print(f'----- Using only NREM (2 and 3) data -----')
        if ADD_SLEEP_STAGE:
            print(colored('Warning: add sleep stage only on NREM!', 'yellow'))
        data, labels, infos_df = select_sleep_stage(data, labels, infos_df, 
                                    stage = [2,3])

    # Nb_channels and frames
    n_channels = data.shape[3]
    nb_frames = data.shape[2]
    n_classes = len(np.unique(labels))
    print('\n-----------------------------------------------')
    print('Number of channels per each frame:', n_channels)
    print('Number of frames in each video:', nb_frames)
    print('Number of classes:', n_classes)
    print('-----------------------------------------------\n')

    # Dataframe with the settings
    param_df = pd.DataFrame(columns=['Parameter', 'Value'])
    param_df = param_df.append({'Parameter' : 'data_filename',
                    'Value'     :  images_filename},          ignore_index=True)      
    param_df = param_df.append({'Parameter' : 'VIDEO_SIZE',
                    'Value'     :  VIDEO_SIZE},               ignore_index=True)  
    param_df = param_df.append({'Parameter' : 'SLIDE',
                    'Value'     :  SLIDE},                    ignore_index=True)  
    param_df = param_df.append({'Parameter' : 'subject_IDs',
                    'Value'     :  subject_IDs},              ignore_index=True)   
    param_df = param_df.append({'Parameter' : 'MODEL_TYPE',
                    'Value'     :  MODEL_TYPE},               ignore_index=True)
    param_df = param_df.append({'Parameter' : 'SUBSAMPLING_TRAIN',
                    'Value'     :  SUBSAMPLING_TRAIN},        ignore_index=True)
    param_df = param_df.append({'Parameter' : 'SUBSAMPLING_VALIDATION',
                    'Value'     :  SUBSAMPLING_VALIDATION},   ignore_index=True)
    param_df = param_df.append({'Parameter' : 'LOSS WEIGHTS',
                    'Value'     :  LOSS_WEIGHTS},             ignore_index=True)                    
    param_df = param_df.append({'Parameter' : 'CONVERT_LABELS',
                    'Value'     :  CONVERT_LABELS},           ignore_index=True)
    param_df = param_df.append({'Parameter' : 'ONLY_LAST_VIDEOS_TRAIN',
                    'Value'     :  ONLY_LAST_VIDEOS_TRAIN},   ignore_index=True)
    param_df = param_df.append({'Parameter' : 'ONLY_LAST_VIDEOS_VAL',
                    'Value'     :  ONLY_LAST_VIDEOS_VAL},     ignore_index=True)
    if ONLY_LAST_VIDEOS_VAL or ONLY_LAST_VIDEOS_TRAIN:
        param_df = param_df.append({'Parameter' : 'N. LAST VIDEOS (%)',
                    'Value'     :  ratio_last_videos},        ignore_index=True)
    param_df = param_df.append({'Parameter' : 'AUGMENT_DATA',
                    'Value'     :  AUGMENT_DATA},             ignore_index=True)
    if AUGMENT_DATA:
        param_df = param_df.append({'Parameter' : 'hor_translations',
                    'Value'     :  hor_tr},                   ignore_index=True)
        param_df = param_df.append({'Parameter' : 'ver_translations',
                    'Value'     :  ver_tr},                   ignore_index=True)
        param_df = param_df.append({'Parameter' : 'noiseSTD',
                    'Value'     :  noiseSTD},                 ignore_index=True)
    param_df = param_df.append({'Parameter' : 'ADD_SLEEP_STAGE',
                    'Value'     :  ADD_SLEEP_STAGE},          ignore_index=True)    
    if ADD_SLEEP_STAGE:
        param_df = param_df.append({'Parameter' : 'SLEEP_STAGE_METHOD',
                    'Value'     :  SLEEP_STAGE_METHOD},       ignore_index=True)
    param_df = param_df.append({'Parameter' : 'USE_ONLY_NREM',
                    'Value'     :  USE_ONLY_NREM},            ignore_index=True)     

    # Save dataframe with the parameters
    param_df.to_pickle(RESULTS_DIR + 'dataframe_summary.pkl')
    print('Dataframe saved in ', RESULTS_DIR)

    # Cross validation
    for subject_ID in subject_IDs:

        print(colored(f'\n----- Leaving out subject {subject_ID} -----\n','green'))

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
        
        # Split validation into test and training
        print('\nSplitting validation into validation (50%) and test (50%)...') 
        data_val, data_test, labels_val, labels_test = split_data(data_val,  
                                    labels_val, 0.5, seed=1, keep_proportions=1)

        # Subsampling the labels
        if SUBSAMPLING_TRAIN:
            print('\nSubsampling train...')
            data_train, labels_train = subsampling_labels(data_train, labels_train, shuffle = True, seed = 1)
        else:
            print('\nLabels in the training set:',np.bincount(labels_train.astype(int)))
        if SUBSAMPLING_VALIDATION:
            print('Subsampling validation...')
            data_val, labels_val = subsampling_labels(data_val, labels_val, shuffle = True, seed = 1)
        else:
            print('Labels in the validation set:', np.bincount(labels_val.astype(int)))
        
        # Take last videos in training and validation
        if ONLY_LAST_VIDEOS_TRAIN:
            print(f'Taking last {ratio_last_videos*100}% videos in the training set...')
            N_videos = int(data_train.shape[1] * ratio_last_videos)
            data_train = data_train[: , -N_videos :]
        if ONLY_LAST_VIDEOS_VAL:
            print(f'Taking last {ratio_last_videos*100}% videos in the validation set...')
            N_videos = int(data_val.shape[1] * ratio_last_videos)
            data_val = data_val[: , -N_videos :]
     
        # Reformat data (no more separated in awakenings)
        x_train, y_train = reformat_data_labels(data_train, labels_train)
        x_val, y_val = reformat_data_labels(data_val, labels_val)

        # Augment data for training set
        if AUGMENT_DATA:
            print(colored('Augmenting data...', 'yellow'))
            x_train, y_train = augment_all_videos(x_train, y_train, 
                                         hor_tr, ver_tr, noiseSTD)        

        # Data loaders
        loader_train = dataloader_pytorch(x_train, y_train, mini_batch_size = 32)
        loader_val = dataloader_pytorch(x_val, y_val, mini_batch_size = y_val.shape[0]//4)

        print('Number of training points:',x_train.shape[0])

        # Start the grid search
        SAVE_WEIGHTS = True
        weights_dir = RESULTS_DIR + f'weights{subject_ID}/'
        if SAVE_WEIGHTS:
            try:
                os.mkdir(weights_dir)
            except:
                print(colored(f'Warning: {weights_dir} already exists...', 'yellow'))
        resultsdf = grid_search_torch(loader_train, loader_val, 
                            n_epoch = 150, 
                            model_type = MODEL_TYPE, 
                            GPU = GPU,
                            n_channels = n_channels, 
                            n_classes = n_classes, 
                            n_frames =  nb_frames,
                            loss_weights = LOSS_WEIGHTS,
                            save_weights = SAVE_WEIGHTS,
                            output_dir = weights_dir )
            
        # Save a dataframe for each subject with grid search results                   
        df_filename = RESULTS_DIR + f'grid_search_results_{subject_ID}.pkl'
        resultsdf.to_pickle(df_filename)
        print(f'Dataframe saved in: {df_filename}' )
    





    








