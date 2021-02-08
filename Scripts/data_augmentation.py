################################################################################
# Here we put all the functions needed to perform data augmentation            #
################################################################################

import numpy as np

def ver_shift(img, trasl, W = 32, H = 32):
    """ Vertical shifting of an image"""
    out = np.zeros(img.shape)
    for i in range(H):
        for j in range(W):
            if ( i + trasl < H and i + trasl >= 0):
                out[i,j] = img[i+trasl,j]
    return out


def hor_shift(img, trasl, W = 32, H = 32):
    """ Horizontal shifting of an image """
    out = np.zeros(img.shape)
    for i in range(H):
        for j in range(W):
            if ( j - trasl < H and j - trasl >= 0):
                out[i,j] = img[i,j-trasl]
    return out


def video_augmentation(video, hor_tr, ver_tr, noiseSTD):
    """Function to augment one single video. All the frames of the videos will be 
    translated in the same way. The added noise, instead, is randomly distributed
    also across time.
       
    Noise is not added where the images are 0.
       
    Video must be of of shape (frames, width, height, channels)
    """
    out = np.copy(video)
    
    # Add noise (only where the images are not zero)
    # Doing so we don't consider the borders. Some other pixel values may be zero,
    # but we assume that they are "few enough"
    if noiseSTD != 0:
        noise = np.random.normal(0, noiseSTD, size = video.shape)
        out[out!=0] = out[out!=0] + noise[out!=0]

    
    # Translation
    # NOTE: maybe the only translation that makes sense is horizontal of 1/-1
    N_frames = out.shape[0]
    width = out.shape[1]
    height = out.shape[2]

    for frame in range(N_frames):
        out[frame] = ver_shift(out[frame], ver_tr, width, height)
        out[frame] = hor_shift(out[frame], hor_tr, width, height)
        
    return out
    

def augment_all_videos(videos, labels, hor_tr = [0], ver_tr = [0], noiseSTD = 0):
    """Return an augmented version of the input videos
        
    Videos is of shape (n_videos, frames, width, height, channels)
    hor_tr and ver_tr are lists that contains the transformations to be applied
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
        for ht in hor_tr:
            for vt in ver_tr:
                v_augmented.append( video_augmentation(vid, ht, vt, noiseSTD) )
                labels_augmented.append(labels[i])
    print('\n')
    return np.array(v_augmented), np.array(labels_augmented)

    
