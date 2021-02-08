import numpy as np
import pandas as pd
from termcolor import colored
import sys

################################################################################
#####################    FUNCTIONS FOR 'RAW-DATA'    ###########################
################################################################################

def add_feature_ratio_to_raw_data(input_data, idx_numerator = 0,
                                                            idx_denumerator = 1):
    '''Function to add ratio of two features as a new feature. This function 
    is meant to be used on 'raw data', meaning data that have still to be cast into
    images and videos.
    Input data, hence, should have the shape (n_awakenings, n_timestep, 256*n_features).
    
    Parameters
    ----------
    input_data : numpy array 
        Data. Format should be (n_awakenings, n_timestep, 256*n_features)
    idx_numerator : int, optional
        Feature to consider as numerator
    idx_denumerator : int, optional
        Feature to consider as denumerator
    
    Returns
    -------
    augmented_data : numpy array
        data augmented (with ratio idx_numerator/idx_denumerator)
        
    '''
    # Check how many features we have and if the data is properly reshaped
    n_features = input_data.shape[2] // 256
    assert( input_data.shape[2] % 256 == 0)
    assert( len(input_data.shape) == 3)
    
    # Check that the numerator idx makes sense
    assert(idx_numerator + 1 <= n_features)
    # Check that the denumerator idx makes sense
    assert(idx_denumerator + 1 <= n_features)
    # Check we are not adding all ones
    assert(idx_denumerator != idx_numerator )
    
    # Take the numerator and the denumerator
    feat_num = input_data[:,:,idx_numerator*256:(idx_numerator+1)*256]
    feat_den = input_data[:,:,idx_denumerator*256:(idx_denumerator+1)*256]
    
    # Check there are no 0 values in the denumerator
    assert(np.sum(feat_den == 0) == 0)
    
    # Create an empty array to store augmented data
    augmented_data = np.empty(( input_data.shape[0], input_data.shape[1], 
                              256 * (n_features + 1) ))
     
    # Fill it
    augmented_data[:,:, 0 : 256*n_features] = input_data;
    augmented_data[:,:, 256*n_features : 256*(n_features+1)] = feat_num/feat_den
    print(f'Data augmented. Added ratio of feature {idx_numerator} ' + \
         f'over feature {idx_denumerator}.')

    return augmented_data




################################################################################
######################    FUNCTIONS FOR 'VIDEOS'    ############################
################################################################################

def add_sleep_stage_single_awakening(awakening, is_REM, method = 'add'):
    '''Add sleep stage to a single video.
    
    Parameters
    ----------
    awakening : numpy array 
        Data to augment. Format should be (videos * frames * channels * heigth * width)
    info_df : pandas dataframe
        Dataframe containing the information of the awakenings.
    method : str, optional
        add: add 2 constant frames, hot-encoding of the sleep stage (REM/NREM)
        multiply: create the constant channels with hot-encoding of the sleep 
                stages and multiply them for the original channels. 
        The new number of channles will duplicate.
    
    Returns
    -------
    augmented_awakening : numpy array
        videos augmented from this awakening
    '''
    n_videos = awakening.shape[0]
    n_frames = awakening.shape[1]
    n_channels = awakening.shape[2]
    H = awakening.shape[3]
    W = awakening.shape[4]
    
    is_NREM = int(is_REM != 1)

    if method == 'add':
        # Add 2 frames representing REM/NREM hot-encoded
        augmented_awakening = np.empty( (n_videos, n_frames, n_channels + 2, H, W) )
        
        for (idx_vid, video) in enumerate(awakening):
            aug_video = np.empty( (n_frames, n_channels + 2, H, W) )

            for frame in range(n_frames):
                aug_video[frame, 0 : n_channels] = video[frame, 0 : n_channels]
                aug_video[frame, n_channels + 0] = is_REM * np.ones((H,W))
                aug_video[frame, n_channels + 1] = is_NREM * np.ones((H,W))
                
            augmented_awakening[idx_vid] = aug_video
    
    elif method == 'multiply':
        # Duplicate the number of frames and multiply them for REM/NREM hot-encoded
        augmented_awakening = np.empty( (n_videos, n_frames, 2*n_channels, H, W) )
        
        for (idx_vid, video) in enumerate(awakening):
            aug_video = np.empty( (n_frames, 2*n_channels, H, W) )

            for frame in range(n_frames):
                for chan in range(n_channels):
                    aug_video[frame, chan] = video[frame, chan] * is_REM
                    aug_video[frame, chan + n_channels] = video[frame, chan] * is_NREM
                    
                    
            augmented_awakening[idx_vid] = aug_video        
    else:
        print(colored(f'Method {method} not recognized', 'red'))
        sys.exit()
            
    return augmented_awakening


def add_sleep_stage_to_all_videos(videos, info_df, method = 'add'):
    '''Add sleep stage to our data.
    
    Parameters
    ----------
    videos : numpy array 
        Data. Format should be (awakenings * videos * frames * channels * length * width)
    info_df : pandas dataframe
        Dataframe containing the information of the awakenings.
    method : str, optional
        add: add 2 constant frames, hot-encoding of the sleep stage (REM/NREM)
        multiply: create the constant channels with hot-encoding of the 
           sleep stages and multiply them for the original channels. 
        The new number of channles will duplicate.
    
    Returns
    -------
    augmented_videos: numpy array
    
    '''
    # Take stages
    sleep_stages = np.array(info_df['Stage'].tolist())
    
    # Check data consistency
    num_awakenings = videos.shape[0]
    assert(sleep_stages.shape[0] == num_awakenings)
    
    augmented_videos = []
    
    for (i, aw) in enumerate(videos):
        if (sleep_stages[i]<=1):
            print(colored(f'WARNING: NREM1 is present at awakening {i+1}', 'yellow'))
        is_REM = int((sleep_stages[i] == 4))
        print(f'Adding sleep stage to awakening {i+1}/{num_awakenings}', end = '\r')
        augmented_videos.append(add_sleep_stage_single_awakening(aw, is_REM, method))
    
    print('\n')
    return np.asarray(augmented_videos)
        
    

##########################################################################################
# NOTE: the following function was straightforward to implement, once the above sleep stage
# function have been implemented. However, in our application the ratio of 2 features
# have been added in the initial phase, i.e using the function add_feature_ratio_to_raw_data
# on the time-series data
##########################################################################################
def add_feature_ratio_single_awakening(awakening, channel_num, channel_den):
    '''Add a new channel to each frame containing the feature ratio 
        channel_num/channel_den.
    
    Parameters
    ----------
    awakening : numpy array 
        Data. Format should be (videos * frames * channels * heigth * width)
    channel_num : int
        Channel to consider as numerator
    channel_den : int
        Channel to consider as denumerator
    
    Returns
    -------
    augmented_awakening : numpy array
        videos augmented from this awakening
    
    '''
    n_videos = awakening.shape[0]
    n_frames = awakening.shape[1]
    n_channels = awakening.shape[2]
    H = awakening.shape[3]
    W = awakening.shape[4]
    

    # Add 1 frame representing channel_num/channel_den
    augmented_awakening = np.empty( (n_videos, n_frames, n_channels + 1, H, W) )
        
    for (idx_vid, video) in enumerate(awakening):
        aug_video = np.empty( (n_frames, n_channels + 1, H, W) )

        for frame in range(n_frames):
            aug_video[frame, 0 : n_channels] = video[frame, 0 : n_channels]
            aug_video[frame, n_channels ] = video[frame, channel_num] / (video[frame, channel_den] + 1e-6) 

                
        augmented_awakening[idx_vid] = aug_video
            
    return augmented_awakening


def add_feature_ratio_to_all_videos(videos, channel_num, channel_den):
    '''Add a new channel to each frame containing channel_num/channel_den.
    
    Parameters
    ----------
    videos : numpy array 
        Data. Format should be (awakenings * videos * frames * channels * length * width)
    channel_num : int
        Channel to consider as numerator
    channel_den : int
        Channel to consider as denumerator
    
    Returns
    -------
    augmented_videos: numpy array
    '''
    num_awakenings = videos.shape[0]
    augmented_videos = []
    
    for (i, aw) in enumerate(videos):
        print(f'Adding ratio of channel {channel_num} over {channel_den} to awakening {i+1}/{num_awakenings}', end = '\r')
        augmented_videos.append(add_feature_ratio_single_awakening(aw, channel_num, channel_den))
    
    print('\n')    
    return np.asarray(augmented_videos)
        
    
    