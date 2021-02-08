import warnings
warnings.filterwarnings("ignore")

import os, sys
import numpy as np
import pandas as pd
import scipy.io as sio


from config import DATA_DIR, MANIPULATED_DATA_DIR
from dreamUtils import *



def create_images(input_filename, output_filename, image_size, normalize = False):
    """
    this function is used to generate images for all awakenings from the raw data.
    It requires us to make sure that the raw data was saved in fold 'MANIPULATED_DATA_DIR'.
    
    :param input_filename: the name of input
    :param output_filename: the name of output
    :image_size: the size of images that we want to generate.
    
    
    :output: the results were saved as the 'output_filename' in MANIPULATED_DATA_DIR directory
    
    """

    #load data
    filename = MANIPULATED_DATA_DIR + input_filename    
    loaded = np.load(filename+'.npz')
    
    data = loaded['data']
    label = loaded['labels']

    print(f'Data loaded from {filename}.npz')
    print('Data shape:',data.shape)
    print('N. features for each electrode:', data.shape[2] // 256)

    #generate images for all awakenings:
    raw_images=[]
    for i in range(data.shape[0]):
        awakening = data[i]
        print(f'Processing awakening: {i+1}/{data.shape[0]}', end='\r')
        images = gen_images(locs_2D, awakening, image_size, normalize=normalize,
                   augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False,
                   verbose = 0)
        raw_images.append(images)
    raw_images = np.array(raw_images)    

    # Save results
    fileName =  MANIPULATED_DATA_DIR + \
                str(image_size)+'_'+str(image_size)+ '_' + output_filename
    np.savez_compressed(fileName, images = raw_images, labels = label )
    print(f'\nSaved in {fileName}.npz')
    
    
def create_video(input_filename, video_size, slide):
    '''
    This function is used to generate videos from images.
    
    :param video_size
    :param slide
    
    :output: the results were saved as the output_filename
    '''
    
    #load data (the generated images)
    print('Loading data from', input_filename )  
    loaded = np.load(input_filename)
    
    data = loaded['images']
    label = loaded['labels']

    #generate videos
    print(f'Creating videos from images with video_size = {video_size} and slide = {slide}.')
    videos_data=[]
    for i in range(data.shape[0]):
        awakening = data[i]
        print(f'Processing awakening: {i+1}/{data.shape[0]}', end='\r')
        Videos = gen_video_from_images(awakening, video_size, slide)
        videos_data.append(Videos)
    print ('\n')
    Videos_data = np.array(videos_data)

    return Videos_data, label


def create_and_save_videos(input_filename, output_filename, video_size, slide):
    '''
    This function is used to generate videos from images.
    
    :param input_filename:
    :param output_filename:
    :param video_size
    :param slide
    
    :output: the results were saved as the output_filename
    '''

    label, Videos_data = create_video(input_filename, video_size, slide)

    # Save results
    print('\nSaving... ')
    fileName =  MANIPULATED_DATA_DIR + \
                str(video_size)+'_'+str(slide)+ '_' + output_filename
    np.savez_compressed(fileName, videos = Videos_data, labels = label )
    print(f'Saved in {fileName}.npz')   


    
if __name__ == '__main__':       
    #generate images for raw data
    create_video('32_32_FFT_SW_log_z-score','FFT_SW_videos_log_z-score', 50, 5)



    


