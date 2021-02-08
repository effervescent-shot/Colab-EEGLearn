import numpy as np
import pandas as pd
import scipy.io as sio


import sys
from dreamUtils import *
from data_augmentation import *
from split_reformat_data import *
from helpers import *
from create_images_videos import *

# pytorch stuff
sys.path.append('decoding_videos/')
from utils_pytorch import *
from pytorch_models import *
from training_with_pytorch import *

if __name__ == '__main__':
    
    GPU = ask_user_integer('Which GPU do you want to use? (type -1 for cpu) ')
    
    SEED = 999
    
    MODEL_TYPE = 'LSTM' #'mix' 2Dconv_max' 'convpool_conv1d' 'LSTM'
    
    RESULTS_DIR = '../Results/Predict_subjects/'  + MODEL_TYPE + f'_SEED{SEED}' +'/'
    
    # Select the filename of the images (without extension)
    images_filename = MANIPULATED_DATA_DIR +  '32_32_FFT_withFeatRatio_log_z-score.npz'

    # Select video parameters
    VIDEO_SIZE = 10
    SLIDE = 3
    
    # Check that the folder where the results are saved does not exists yet
    try:
        os.mkdir(RESULTS_DIR)
    except:
        print(colored(f'Directory {RESULTS_DIR} already exists!', 'red'))
        continue_bool = ask_user_0_1('Do you still want to continue? (0=NO/1=YES) ', print_0 = 'Exit.')
        if continue_bool == 0 :   
            exit()

    # Generate videos
    data, labels = create_video(images_filename, VIDEO_SIZE, SLIDE)
    labels = labels.flatten()

    # Load dataframe with infos
    filename = MANIPULATED_DATA_DIR  + 'awakenings_info_df_FFT_SW.pkl'
    print('Loading infos dataframe from:', filename)
    infos_df = pd.read_pickle(filename)

    # Set index and add name_code column
    infos_df = infos_df.set_index(np.arange(len(infos_df)))
    infos_df['name_code'] = infos_df.apply(lambda row: row['Subject_id']+ '_' + row['Quest_number'], axis=1)

    subject_IDs = ['H009', 'H018', 'H019', 'H021', 'H026', 'H025', 'H033', 'H035',
           'H048', 'H050', 'H051', 'H054', 'H055', 'H057', 'H060', 'H061']  

    infos_df['subject_code'] = np.ones(data.shape[0],int)

    label = -1
    for subject_ID in subject_IDs:
        label += 1
        temp = infos_df[infos_df['Subject_id'].str.contains(subject_ID)]
        infos_df['subject_code'][temp.index] = label    

    # assign labels as subjects
    label = infos_df.subject_code.values

    X_train = np.empty((0,data.shape[1], data.shape[2], data.shape[3], data.shape[4], data.shape[5]))
    X_val = np.empty((0,data.shape[1], data.shape[2], data.shape[3], data.shape[4], data.shape[5]))
    Y_train = np.empty((0))
    Y_val = np.empty((0))

    for i in np.unique(label):
        y = label[np.where(label == i)]
        x = data[np.where(label == i)]

        # for each subject, split data to train/validation (0.8/0.2)
        x_tr, x_val, y_tr, y_val = split_data(x, y, 0.8, seed = SEED)

        X_train = np.vstack((X_train, x_tr))
        X_val = np.vstack((X_val, x_val))
        Y_train = np.hstack((Y_train, y_tr))
        Y_val = np.hstack((Y_val, y_val))

    # Reformat data (no more separated in awakenings)
    X_train, Y_train = reformat_data_labels(X_train, Y_train)
    X_val, Y_val = reformat_data_labels(X_val, Y_val)

    # Data loaders
    loader_train = dataloader_pytorch(X_train, Y_train, mini_batch_size = 32)
    loader_val = dataloader_pytorch(X_val, Y_val, mini_batch_size = 32)

    n_classes = len(subject_IDs)
    nb_frames = X_train.shape[1]
    n_channels = X_train.shape[-1]

    SAVE_WEIGHTS = False
    weights_dir = RESULTS_DIR
    if SAVE_WEIGHTS:
        try:
            os.mkdir(weights_dir)
        except:
            print(colored(f'Warning: {weights_dir} already exists...', 'yellow'))
    resultsdf = grid_search_torch(loader_train, loader_val, 
                                n_epoch = 100, 
                                model_type = MODEL_TYPE, 
                                GPU = GPU,
                                n_channels = n_channels, 
                                n_classes = n_classes, 
                                n_frames =  nb_frames,
                                save_weights=SAVE_WEIGHTS,
                                output_dir=weights_dir)
    df_filename = RESULTS_DIR + 'results.pkl'
    resultsdf.to_pickle(df_filename)
    print(f'Dataframe saved in: {df_filename}' )
    
       