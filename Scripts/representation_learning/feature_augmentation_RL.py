import numpy as np
import pandas as pd
from termcolor import colored
import sys

################################################################################
######################    FUNCTIONS FOR 'IMAGES'    ############################
################################################################################

def add_sleep_stage_single_awakening_images(awakening, is_REM, method = 'add'):
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
    n_images = awakening.shape[0]
    n_channels = awakening.shape[1]
    H = awakening.shape[2]
    W = awakening.shape[3]
    
    is_NREM = int(is_REM != 1)

    if method == 'add':
        # Add 2 frames representing REM/NREM hot-encoded
        augmented_awakening = np.empty( (n_images, n_channels + 2, H, W) )
        
        for (idx_vid, image) in enumerate(awakening):
            aug_image= np.empty( ( n_channels + 2, H, W) )

            aug_image[0 : n_channels] = image[0 : n_channels]
            aug_image[n_channels + 0] = is_REM * np.ones((H,W))
            aug_image[n_channels + 1] = is_NREM * np.ones((H,W))
                
            augmented_awakening[idx_vid] = aug_image
    
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


def add_sleep_stage_to_all_images(images, info_df, method = 'add'):
    '''Add sleep stage to our data.
    
    Parameters
    ----------
    videos : numpy array 
        Data. Format should be (awakenings * frames * channels * length * width)
    info_df : pandas dataframe
        Dataframe containing the information of the awakenings.
    method : str, optional
        add: add 2 constant frames, hot-encoding of the sleep stage (REM/NREM)
        multiply: create the constant channels with hot-encoding of the 
           sleep stages and multiply them for the original channels. 
        The new number of channles will duplicate.
    
    Returns
    -------
    augmented_images: numpy array
    
    '''
    # Take stages
    sleep_stages = np.array(info_df['Stage'].tolist())
    
    # Check data consistency
    num_awakenings = images.shape[0]
    assert(sleep_stages.shape[0] == num_awakenings)
    
    augmented_images = []
    
    for (i, aw) in enumerate(images):
        if (sleep_stages[i]<=1):
            print(colored(f'WARNING: NREM1 is present at awakening {i+1}', 'yellow'))
        is_REM = int((sleep_stages[i] == 4))
        print(f'Adding sleep stage to awakening {i+1}/{num_awakenings}', end = '\r')
        augmented_images.append(add_sleep_stage_single_awakening_images(aw, is_REM, method))
    
    print('\n')
    return np.asarray(augmented_images)