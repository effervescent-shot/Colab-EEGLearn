import sys
sys.path.append('../')
from dreamUtils import *
from data_augmentation import *
from split_reformat_data import *
from helpers import *
from feature_augmentation_RL import *
from config import *

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import OneHotEncoder

# pytorch stuff
from util_RL import *
from pytorch_models_RL import *
from training_with_pytorch_RL import *
from utils_pytorch_RL import *


# This file is used to train the representation learning and generate the encoder 
# for next decoding part in leave-one-subject-out cross-validation way. We further 
# devided validation set to validation and test. We train the representation learning
# on train set and test on validation set. we use early-stopping by computing the mse loss.
# We leave the test set untouched on this stage.
# We saved our models and the encoders (which is the encoder vectors outputted by the trained models).

# data: Video
# labels: image (the one used as representation learning), class (the one we want to predict), sleep stage.
# RL models: take video and image, we use the output the representation learning vector (encoder).

# output:
# dataframe: contains all the information of the grid search.
# 'weight{subject}' folder: return the best models after early stopping for all grid search.
#                           return the encoders (in zip file) that generate from encoder.
#                           (inside the zip file: 'x_train_encoder', 'x_val_encoder','x_test_encoder',
#                           'label_classes_train', 'label_classes_val', 'label_classes_test',
#                            'sleep_stage_train','sleep_stage_val','sleep_stage_test')


if __name__ == '__main__':
    MANIPULATED_DATA_DIR = '../' +  MANIPULATED_DATA_DIR
    
    # Select the GPU (at runtime)
    GPU = ask_user_integer('Which GPU do you want to use? (type -1 for cpu) ')

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> change the parameter<<<<<<<<<<<<<<<<<<<<
    # choose how many frame after the video is the start of your representation label
    N_after_videos = 4 # default is 1, which is the one immediately after the video

    # choose how many frames you wanna predict
    n_predicted_frames = 1
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> change the parameter<<<<<<<<<<<<<<<<<<<<
    # Which model?
    MODEL_TYPE = mix #'2Dconv_max' 'mix'  'convpool_conv1d' 'LSTM'  '2Dconv_max'
                       
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> change the parameter<<<<<<<<<<<<<<<<<<<<
    # the ratio and the random seed of spilting validation/test
    ratio_val = 0.5
    seed_val = 2
    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> decide if add sleep stage<<<<<<<<<<<<<<<<<<<<
    ADD_SLEEP_STAGE = 1

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> change the parameter<<<<<<<<<<<<<<<<<<<<
    # Select folder name, where the reults will be saved
    RESULTS_DIR = '../../Results/' + MODEL_TYPE + '_' + str(n_predicted_frames) + 'frames' + '_seed_val' + str(seed_val) + '_ratio_val' + str(ratio_val) + '/'
    # Check that the folder where the results are saved does not exists yet
    try:
        os.mkdir(RESULTS_DIR)
    except:
        print(colored(f'Directory {RESULTS_DIR} already exists!', 'red'))
        continue_bool = ask_user_0_1('Do you still want to continue? (0=NO/1=YES) ', print_0 = 'Exit.')
        if continue_bool == 0 :   
            exit()

    # Select the filename of the images (without extension)
    images_filename = MANIPULATED_DATA_DIR + '32_32_FFT_withFeatRatio_log_z-score.npz'
    images = np.load(images_filename)
    data = images['images']
    labels = images['labels']  
    label_classes = labels.flatten()

    # Select the subjects to be used for the cross validation
    subject_IDs = ['H009', 'H018', 'H019', 'H021', 'H026', 'H025', 'H033', 'H035', 'H048', 'H050', 'H051', 'H054', 'H055', 'H057', 'H060','H061']

    # Load dataframe with infos
    df_filename = MANIPULATED_DATA_DIR + 'awakenings_info_df_FFT_SW.pkl'
    print('Loading infos dataframe from:', df_filename)
    infos_df = pd.read_pickle(df_filename)
    # Set index and add name_code column
    infos_df = infos_df.set_index(np.arange(len(infos_df)))
    infos_df['name_code'] = infos_df.apply(lambda row: row['Subject_id']+ '_' + row['Quest_number'], axis=1)
    
    # get the information of subject code
    label_subjects = infos_df['Subject_id'].apply(lambda x: x[0:4])
    label_subjects = label_subjects.values
    
    # Add sleep stage as a feature
    if ADD_SLEEP_STAGE:
        print(f'----- Adding REM/NREM feature  -----')
        data = add_sleep_stage_to_all_images(data, infos_df)
    
    
    # RAM/NORAM
    sleep_stage = infos_df['Stage'].values

    #generate the video the predicted frame and the class labels
    video_size = 10
    slide = 3  
    print(data.shape)
    x_videos, label_images = gen_data_for_representation_learning(data,n_predicted_frames, video_size, slide,N_after_videos)


    
    
    # # convert label
    # # conver class labels: DE, DEWR --> DE
    # label_classes[label_classes == 2] =1
	# # RAM 1, NRAM = 0
    # sleep_stage[sleep_stage != 4] = 0
    # sleep_stage[sleep_stage == 4] = 1

    # Nb_channels and frames
    n_channels = x_videos.shape[3]
    n_frames = video_size
    
    


# Cross validation
    for subject_ID in subject_IDs:

        print(colored(f'\n----- Leaving out subject {subject_ID} -----\n','green'))

        # build the foler to save the weights and encoder data
        weights_dir = RESULTS_DIR + f'weights{subject_ID}/'
        try:
            os.mkdir(weights_dir)
        except:
            print(colored(f'Warning: {weights_dir} already exists...', 'yellow'))

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
        data_train,label_images_train, label_classes_train,label_subjects_train,sleep_stage_train, data_val, label_images_val, label_classes_val, label_subjects_val,sleep_stage_val = split_train_test_RL(x_videos, 
                label_images,label_classes,label_subjects,sleep_stage, infos_df, test_codes)

        # Split validation into validation and test
        print(label_subjects_val)
        print(label_classes_val)
        print(f'\nSplitting validation into validation with ratio {ratio_val}% for validation...') 
        data_val, data_test, label_images_val, label_images_test,\
        label_classes_val, label_classes_test, label_subjects_val,label_subjects_test,sleep_stage_val, sleep_stage_test = \
            split_data_RL(data_val, label_images_val, label_classes_val,label_subjects_val, sleep_stage_val, ratio_val, seed=seed_val)

        # Reformat data (merge the awakings)
        x_train, y_train,label_classes_train, label_subjects_train,sleep_stage_train= reformat_data_labels_RL(data_train, label_images_train,label_classes_train,label_subjects_train,sleep_stage_train)
        x_val, y_val,label_classes_val,label_subjects_val,sleep_stage_val = reformat_data_labels_RL(data_val, label_images_val,label_classes_val,label_subjects_val,sleep_stage_val)
        x_test, y_test,label_classes_test,label_subjects_test,sleep_stage_test = reformat_data_labels_RL(data_test, label_images_test,label_classes_test,label_subjects_test,sleep_stage_test)
        
        ####################### genterate data for the deocder part #####################
        #  classes label
        label_classes_train = torch.from_numpy(label_classes_train)
        label_classes_train = label_classes_train.type(torch.LongTensor)
        label_classes_val = torch.from_numpy(label_classes_val)
        label_classes_val = label_classes_val.type(torch.LongTensor)   
        label_classes_test = torch.from_numpy(label_classes_test)
        label_classes_test = label_classes_test.type(torch.LongTensor)  
        
        # sleep stage label
        sleep_stage_train = torch.from_numpy(sleep_stage_train)
        sleep_stage_train = sleep_stage_train.type(torch.float)
        sleep_stage_val = torch.from_numpy(sleep_stage_val)
        sleep_stage_val = sleep_stage_val.type(torch.float) 
        sleep_stage_test = torch.from_numpy(sleep_stage_test)
        sleep_stage_test = sleep_stage_test.type(torch.float) 

        #################### Start the representation learning ##############         
        # Data loaders for representation learning part
        loader_train = dataloader_pytorch_RL(x_train, y_train, mini_batch_size = 32,shuffle=True)
        loader_val = dataloader_pytorch_RL(x_val, y_val, mini_batch_size = y_val.shape[0],shuffle=True)

        # Dataloader for encoder part
        loader_train_encoder = dataloader_pytorch_RL(x_train,np.empty(0), mini_batch_size = 32,shuffle=False)
        loader_val_encoder = dataloader_pytorch_RL(x_val,np.empty(0), mini_batch_size = 32,shuffle=False)
        loader_test_encoder = dataloader_pytorch_RL(x_test,np.empty(0), mini_batch_size = 32,shuffle=False)

        # Train representation learning with grid search; save the model, results and encoder for next step.
        resultsdf = representation_learning_training(loader_train, loader_val,
                                loader_train_encoder,loader_val_encoder, loader_test_encoder,
                                label_classes_train,label_classes_val, label_classes_test,
                                label_subjects_train,label_subjects_val, label_subjects_test,
                                sleep_stage_train,sleep_stage_val, sleep_stage_test,
                                n_epoch = 2, model_type = MODEL_TYPE, GPU = GPU,
                                n_channels = n_channels, 
                                n_frames =  n_frames, n_predicted_frames=n_predicted_frames, 
                                patient=10, output_dir = weights_dir)   
                         
        # Save a dataframe for each subject with grid search results                   
        df_filename = RESULTS_DIR + f'grid_search_results_{subject_ID}.pkl'
        resultsdf.to_pickle(df_filename)
        print(f'Dataframe saved in: {df_filename}' )