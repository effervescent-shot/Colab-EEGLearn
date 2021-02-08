################################################################################
# Authors: Nihal Ezgi Yuceturk, Hongyu Luo and Nicola Ischia       (EPFL)      #
#                                                                              #
# Date:                                                                        #
#                                                                              #
# General utils to handle data and its generation                              #
################################################################################


import math as m
import numpy as np
import scipy.io
import scipy.signal
from sklearn.decomposition import PCA
import pandas as pd
from termcolor import colored
import sys

from functools import reduce
import math as m
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
np.random.seed(1235)

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

def centerize_reference(x):
    electrode_means = x.mean(axis=1, keepdims=True)
    x = x-electrode_means
    #print(x.shape)
    return x


def map_to_2d(locs_3D):
    """
    Maps the 3D positions of the electrodes into 2D plane with AEP algorithm 
    
    :param locs_3D: matrix of shape number_of_electrodes x 3, for X,Y,Z coordinates respectively
    :return: matrix of shape number_of_electrodes x 2
    """
    locs_2D = []
    for e in locs_3D:
        locs_2D.append(azim_proj(e))
    
    return np.array(locs_2D)



def gen_images(locs, features, n_gridpoints, normalize=False,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False,
               verbose = 1, interpolation = 'cubic'):
    """
    source : https://github.com/pbashivan/EEGLearn/eeg_cnn_lib.py
    
     Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
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



def gen_video_from_images(images, video_size, slide):
    """
    generate videos from images for each single awakening.
    
    :param images: the image data with the shape (n_timepoints, n_channels, 32, 32)
    :param video_size: the size of frames of video
    :param slide: the size of frames of sliding window
        
    :return: the data of videos. 
    """ 
    num_video = (images.shape[0]-video_size)//slide + 1
    
    start = 0
    x_tr_video=[]
    for i in range(num_video):
        a_video = images[start: start+video_size]
        x_tr_video.append(a_video)
        start += slide
        
    return np.array(x_tr_video)


def gen_data_for_representation_learning(images, labels, video_size, slide):
    """
    generate videos from images for all awakenings.
    
    :param images: the image data with the shape (n_awaking, n_frames, nb_channels, 32, 32)
    :param labels: the label of each awanking (n_awaking, )
    :param video_size: the size of frames of video
    :param slide: the size of frames of sliding window
        
    :return: the data of videos. 
    """
    num_video = (images.shape[1]-video_size)//slide + 1
    print(num_video)
    
    x_tr_video = []
    y_tr_image = []
    new_labels = []
    for (idx, images_aw) in enumerate(images):
        for i in range(num_video):
            a_video = images_aw[i*slide : i*slide+video_size]
            a_image = images_aw[i*slide+video_size].flatten()
            x_tr_video.append(a_video)
            y_tr_image.append(a_image)
            new_labels.append(labels[idx])
        

    
    return np.array(x_tr_video), np.array(y_tr_image), np.array(new_labels)



def downsample_in_time_raw_data(data, factor = 4, use_scipy = False):
    '''Downsampling in time of raw data.
    
    Parameters
    ----------
    data : 3-D numpy array [awakenings x time x eeg]
        Data to be downsampled
    factor : int, optional
        Downsampling factor
    use_scipy : boolean, optional
        Use the function scipy.signal.decimate to downsample
    
    Returns
    -------
    downsampled_data : 3-D numpy array
        Data after the downsampling
    downsampled_length : int
        Length of the time series in the downsampled data
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
            downsampled_awakening = scipy.signal.decimate(awakening,factor,axis=0,
                                                        zero_phase=True)
            downsampled_data[idx] = downsampled_awakening
    else:
        for idx,awakening in enumerate(data):
            downsampled_awakening = downsample_single_awakening(awakening, factor)
            downsampled_data[idx] = downsampled_awakening
        
    print('Done!')
    return downsampled_data, downsampled_length



def downsample_single_awakening(awakening, factor):
    '''Downsampling in time of raw data for a single awakening taking the average 
    of batches of consecutive datapoints in time of dimension 'factor'.
    
    Parameters
    ----------
    awakening : 2-D numpy array [time x eeg]
        Data to be downsampled
    factor : int
        Downsampling factor

    
    Returns
    -------
    downsampled_awakening : 2-D numpy array
        Data after the downsampling
    '''
    N = awakening.shape[0] // factor
    downsampled_awakening = np.empty((N,awakening.shape[1]))
    for i in range(N):
        downsampled_awakening[i] = np.mean(awakening[i*factor:(i+1)*factor],
                                        axis = 0)
    
    return downsampled_awakening
    


def normalize_each_subject(input_data, info_df, log = True, method = 'min-max'):
    '''
    Function to normalize data computing different parameters for each subject. This function is 
    meant to be used on 'raw data', meaning data that have still to be cast into images.
    Input data, hence, should have the shape (n_awakenings, n_timestep, 256*n_features).
    
    Parameters
    ----------
    input_data : numpy array 
        Data. Format should be (n_awakenings, n_timestep, 256*n_features)
    info_df : pandas df
        Dataframe containing information of the awakenings
    log : bool, optional
        Decide to take the log before doing normalization
    method : string, optional
        Type of normalization. Methods available: 'min-max', 'z-score', 'none'
    
    Returns
    -------
    normalized_data : numpy array
        data normalized
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
                    mean_feat = np.mean( subject_data[:,:,feat*256:(feat+1)*256] )
                    std_feat = np.std( subject_data[:,:,feat*256:(feat+1)*256] )
                    normalized_data[subject_index,:,feat*256:(feat+1)*256] = \
                        (subject_data[:,:,feat*256:(feat+1)*256] - mean_feat)/std_feat
                else:
                    print(colored(f'Method {method} not implemented.','red'))
                    sys.exit()
         
    print('Data normalized within subject with method:', method)
        
    return normalized_data


def select_sleep_stage(awakenings, labels, info_df, stage = [4]):
    '''Function to select awakenings in specific sleep stages.
    
    Parameters
    ----------
    awakenings : numpy array 
        Data. Format should be (awakenings, \*whatever*\)
    labels : numpy array
        label of each awakening
    info_df : pandas dataframe
        Dataframe containing the information of the awakenings.
    stage : list of integers, optional
        From 1 to 3: NREM1, NREM2, NREM3. 4: REM
    
    Returns
    -------
    selected_aw : numpy array
        awakenings with the selected sleep stages
    selected_lab : numpy array
        labels of the selected awakenings
    selected_df : pandas dataframe
        dataframe containing the informations of the selected awakenings
    '''
    # Take stages
    sleep_stages = np.array(info_df['Stage'].tolist())
    
    # Check data consistency
    num_awakenings = awakenings.shape[0]
    assert(sleep_stages.shape[0] == num_awakenings)
    
    # Check input makes sense
    for s in stage:
        if s not in [1,2,3,4]:
            print(colored(f'Warning: stage {s} not recognized', 'yellow'))

    selected_aw = []
    selected_lab = []
    selected_df = pd.DataFrame(columns = info_df.columns)
    for (i, aw) in enumerate(awakenings):
        if sleep_stages[i] in stage:
            selected_aw.append(aw)
            selected_df 
            selected_df = selected_df.append(info_df.iloc[i])
            selected_lab.append(labels[i])
            
    selected_df = selected_df.set_index(np.arange(len(selected_df)))
    
    return np.asarray(selected_aw), np.asarray(selected_lab), selected_df
