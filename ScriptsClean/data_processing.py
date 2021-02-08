import sys
import math as m
import numpy as np
import pandas as pd
from functools import reduce
from termcolor import colored

import scipy.io
import scipy.signal
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
np.random.seed(1235)

def shift_vertical(img, shift, W = 32, H = 32):
    """ Vertical shifting of an image"""
    out = np.zeros(img.shape)
    for i in range(H):
        for j in range(W):
            if ( i + shift < H and i + shift >= 0):
                out[i,j] = img[i+shift,j]
    return out


def shift_horizontal(img, shift, W = 32, H = 32):
    """ Horizontal shifting of an image """
    out = np.zeros(img.shape)
    for i in range(H):
        for j in range(W):
            if ( j - shift < H and j - shift >= 0):
                out[i,j] = img[i,j-shift]
    return out


def add_noise_and_shift_single_video(video, hor_shift, ver_shift, noiseSTD):
    """
    Augment single video (multiple frames) with noise. 
    The added noise randomly distributed also across time and space.
    Noise is not added where the images are 0.
       
    Video must be of of shape (frames, width, height, channels)
    """
    out = np.copy(video)
    
    # Add noise (only where the images are not zero)
    # Doing so we don't consider the borders. Some other pixel values may be zero,

    if noiseSTD != 0:
        noise = np.random.normal(0, noiseSTD, size = video.shape)
        out[out!=0] = out[out!=0] + noise[out!=0]

    
    # Translation
    # NOTE: maybe the only translation that makes sense is horizontal of 1/-1
    N_frames = out.shape[0]
    width = out.shape[1]
    height = out.shape[2]

    for frame in range(N_frames):
        out[frame] = shift_vertical(out[frame], ver_shift, width, height)
        out[frame] = shift_horizontal(out[frame], hor_shift, width, height)
        
    return out
    

def add_noise_and_shift_all_videos(videos, labels, hor_shift = [0], ver_shift = [0], noiseSTD = 0):
    """
    Return an augmented version of the input videos
    Videos is of shape (n_videos, frames, width, height, channels)
    :param hor_tr and ver_tr: lists contain amount of shift to be applied +- direction 
    """
    v_augmented = []
    labels_augmented = []
    print('Augmenting data ...')
    for (i, vid) in enumerate(videos):
        if (i+1) % 100 == 0: print(f'Augmenting video {i+1}/{videos.shape[0]}', end="\r" )
        # Keep also the original video
        v_augmented.append(vid)
        labels_augmented.append(labels[i])
        
        # Apply vertical and horizontal translations and combine them
        for ht in hor_shift:
            for vt in ver_shift:
                v_augmented.append(add_noise_and_shift_single_video(vid, ht, vt, noiseSTD) )
                labels_augmented.append(labels[i])
    print('Done!')
    return np.array(v_augmented), np.array(labels_augmented)


def add_feature_ratio_data(input_data, idx_numerator = 0, idx_denumerator = 1):
    '''
    Function to add ratio of two features as a new feature (ratio = idx_numerator/idx_denumerator)
    
    :param input_data (numpy array): daha in shape (n_awakenings, n_timestep, 256*n_features)
    :param idx_numerator  (int): optional, feature to consider as numerator
    :param idx_denumerator (int): optional, feature to consider as denumerator
    :return: data augmented with ratio as numpy array  
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


def add_sleep_stage_single_awakening(awakening, is_REM, method = 'add'):
    '''
    Add sleep stage to a single video.
    
    :param awakening (numpy array): data in shape (video * frames * channels * heigth * width)
    :param info_df (pandas.dataframe): dataframe containing the information of the awakenings.
    :param method (str): optional as 
                         add = add 2 constant frames, hot-encoding of the sleep stage (REM/NREM)
                         multiply= create the constant channels with hot-encoding of the sleep 
                         stages and multiply them for the original channels. Number of channles will duplicate.
    :return: augmented_awakenig videos as numpy array
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


def add_sleep_stage_all_awekenings(videos, info_df, method = 'add'):
    '''
    Add sleep stage to our data.
    
    :param awakening (numpy array): data in shape (awakenings * videos * frames * channels * length * width)
    :param info_df (pandas.dataframe): dataframe containing the information of the awakenings.
    :param method (str): optional as 
                         add = add 2 constant frames, hot-encoding of the sleep stage (REM/NREM)
                         multiply= create the constant channels with hot-encoding of the sleep 
                         stages and multiply them for the original channels. Number of channles will duplicate.
    :return: augmented_videos as numpy array
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


def add_feature_ratio_single_awakening(awakening, channel_num, channel_den):
    '''
    Add a new channel to each frame containing the feature ratio channel_num/channel_den.
    
    :param awakening (numpy array): data in shape (videos * frames * channels * heigth * width)
    :param channel_num (int): channel to consider as numerator
    :param channel_den (int): channel to consider as denumerator
    :return: augmented_awakening videos as numpy array
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


def add_feature_ratio_to_all_awekenings(videos, channel_num, channel_den):
    '''
    Add a new channel to each frame containing channel_num/channel_den.
    
    :param awakening (numpy array): data in shape (awakenings * videos * frames * channels * length * width)
    :param channel_num (int): channel to consider as numerator
    :param channel_den (int): channel to consider as denumerator
    :return: augmented_awakening videos as numpy array
    '''
    num_awakenings = videos.shape[0]
    augmented_videos = []
    
    for (i, aw) in enumerate(videos):
        print(f'Adding ratio of channel {channel_num} over {channel_den} to awakening {i+1}/{num_awakenings}', end = '\r')
        augmented_videos.append(add_feature_ratio_single_awakening(aw, channel_num, channel_den))
    
    print('\n')    
    return np.asarray(augmented_videos)


def downsample_single_awekening_in_time(awakening, factor):
    '''
    Downsampling in time of raw data for a single awakening taking the average 
    of batches of consecutive datapoints in time of dimension 'factor'.
    
    :param awakening: 2-D numpy array [time x eeg], data to be downsampled
    :param factor (int): fownsampling factor
    :return: downsampled_awakening as 2-D numpy array
    '''
    N = awakening.shape[0] // factor
    downsampled_awakening = np.empty((N,awakening.shape[1]))
    for i in range(N):
        downsampled_awakening[i] = np.mean(awakening[i*factor:(i+1)*factor],axis = 0)
    
    return downsampled_awakening


def downsample_data_in_time(data, factor = 4, use_scipy = False):
    '''Downsampling in time of raw data.
   
    :param data: 3-D numpy array [awakenings x time x eeg], data to be downsampled
    :param factor (int): optional, downsampling factor
    :param use_scipy (boolean): optional, use the function scipy.signal.decimate to downsample
    :return: downsampled_data as 3-D numpy array, downsampled_length as length of the time series after
    '''
    # Number of datapoints in each time series of the input data
    original_length = data.shape[1]
    
    # After the downsampling, how many datapoints in each time series?
    if use_scipy:
        downsampled_length = int(np.ceil(original_length // factor))
    else:
        downsampled_length = original_length // factor
    downsampled_data = np.empty((data.shape[0],downsampled_length,data.shape[2]))
    
    print('Downsampling data with a factor of:', factor)
    print('Original data shape of:', data.shape)
    print('Downsampled data shape of:', downsampled_data.shape)
    print(f'Downsampling EEG time series of {data.shape[0]} awakenings...')
    
    if use_scipy:
        for idx,awakening in enumerate(data):
            downsampled_awakening = scipy.signal.decimate(awakening,factor,axis=0, zero_phase=True)
            downsampled_data[idx] = downsampled_awakening
    else:
        for idx,awakening in enumerate(data):
            downsampled_awakening = downsample_single_awekening_in_time(awakening, factor)
            downsampled_data[idx] = downsampled_awakening
        
    print('Done!')
    return downsampled_data, downsampled_length    


def generate_video_from_images(images, video_size, slide):
    """
    Generate videos from images for each single awakening.
    
    :param images: the image data with the shape (n_timepoints, n_channels, 32, 32)
    :param video_size: the size of frames of video
    :param slide: the size of frames of sliding window    
    :return: video from list of frames. 
    """ 
    num_video = (images.shape[0]-video_size)//slide + 1
    
    start = 0
    x_tr_video=[]
    for i in range(num_video):
        a_video = images[start: start+video_size]
        x_tr_video.append(a_video)
        start += slide
      
    return np.array(x_tr_video)


def normalize_by_subject(input_data, info_df, log = True, method = 'min-max'):
    '''
    Function to normalize data computing different parameters for each subject.    

    :param input_data (numpy array): data in shape (n_awakenings, n_timestep, 256*n_features)
    :param info_df (pandas.dataframe): df containing information of the awakenings
    :param log (bool): optional, decide to take the log before doing normalization
    :param method (string): optional, type of normalization 'min-max', 'z-score', 'none'
    :return: normalized_data (numpy array) 
    '''
    # Check how many features we have and if the data is properly reshaped
    n_features = input_data.shape[2] // 256
    assert( input_data.shape[2] % 256 == 0)
    assert( len(input_data.shape) == 3)
    
    print('Data shape:', input_data.shape)
    print('N. features:', n_features)
    
    # Take the subjects identifiers from the dataframe as a list
    info_df = info_df.set_index(np.arange(len(info_df)))
    subject = info_df['Subject_id'].astype(str).str[0:4].unique()
    
    # First (optional) preprocessing step
    if log:
        assert(np.sum(input_data <= 0) == 0) # We don't want data less than 0
        print('Taking the log of input data...')
        normalized_data = np.log(input_data)
    else:
        normalized_data = np.copy(input_data)

    if method == 'none':
        normalized_data = normalized_data
    else:
        # Normalize separately each subject
        for i in subject:
            
            # Take the data related to the idxs of the ith subject
            subject_index = info_df[info_df['Subject_id'].str.contains(i)].index
            subject_data = normalized_data[subject_index]

            # Normalize each feature (through time)
            for feat in range(n_features):

                if method == 'min-max':
                    max_feat = np.max( subject_data[:,:,feat*256:(feat+1)*256] )
                    min_feat = np.min( subject_data[:,:,feat*256:(feat+1)*256] )
                    normalized_data[subject_index,:,feat*256:(feat+1)*256] = \
                        (subject_data[:,:,feat*256:(feat+1)*256] - min_feat) / (max_feat-min_feat)
                        
                elif method == 'z-score':
                    mean_feat = np.mean(subject_data[:,:,feat*256:(feat+1)*256] )
                    std_feat = np.std( subject_data[:,:,feat*256:(feat+1)*256] )
                    normalized_data[subject_index,:,feat*256:(feat+1)*256] = \
                        (subject_data[:,:,feat*256:(feat+1)*256] - mean_feat)/std_feat
                else:
                    print(colored(f'Method {method} not implemented.','red'))
                    sys.exit()
         
    print('Data normalized within subject with method:', method) 
    return normalized_data


def cart2sph(x, y, z):
    """
    source : https://github.com/pbashivan/EEGLearn/utils.py
    
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x**2 + y**2
    r = m.sqrt(x2_y2 + z**2)                    # r
    elev = m.atan2(z, m.sqrt(x2_y2))            # Elevation
    az = m.atan2(y, x)                          # Azimuth
    return r, elev, az


def pol2cart(theta, rho):
    """
    source : https://github.com/pbashivan/EEGLearn/utils.py
    
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    return rho * m.cos(theta), rho * m.sin(theta)

def azim_proj(pos):
    """
    source : https://github.com/pbashivan/EEGLearn/eeg_cnn_lib.py
    
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, m.pi / 2 - elev)


def map_3d_to_2d(locs_3D):
    """
    Maps the 3D positions of the electrodes into 2D plane with AEP algorithm 
    
    :param locs_3D: matrix of shape number_of_electrodes x 3, for X,Y,Z coordinates respectively
    :return: matrix of shape number_of_electrodes x 2
    """
    locs_2D = []
    for e in locs_3D:
        locs_2D.append(azim_proj(e))
    
    return np.array(locs_2D)


def generate_images(locs, features, n_gridpoints, normalize=False,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False,
               verbose = 1, interpolation = 'cubic'):
    """
    source : https://github.com/pbashivan/EEGLearn/eeg_cnn_lib.py
    
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]. Features are as columns.
                                Features corresponding to each frequency band concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:    Flag for whether to normalize each band over all samples
    :param augment:      Flag for generating augmented images
    :param pca:          Flag for PCA based data augmentation
    :param std_mult      Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:     If True generates edgeless images by adding artificial channels
                         at four corners of the image with value = 0 (default=False).
    :return:             Tensor of size [samples, colors, W, H] containing generated images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]     # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    #assert features.shape[1] % nElectrodes == 0
    n_colors = int(features.shape[1] / nElectrodes)
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes : nElectrodes * (c+1)])

    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    # Interpolating
    for i in range(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                    method=interpolation, fill_value=np.nan)
        if verbose: print('Interpolating {0}/{1}'.format(i+1, nSamples), end='\r')
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
        
    if verbose: print('Interpolated {0}/{1} '.format(nSamples, nSamples))
    return np.swapaxes(np.asarray(temp_interp), 0, 1) 


