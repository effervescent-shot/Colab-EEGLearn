import sys
import os
import regex as re
import numpy as np
import pandas as pd
import scipy.io as sio
from config import *
from data_processing import generate_video_from_images, generate_images, map_3d_to_2d


def generate_video_data_for_representation_learning(images, labels, video_size, slide):
    """
    Generate videos from images for all awakenings.
    
    :param images: the image data with the shape (n_awaking, n_frames, nb_channels, 32, 32)
    :param labels: the label of each awanking (n_awaking, )
    :param video_size: the size of frames of video
    :param slide: the size of frames of sliding window
    :return: video frames, target frame, actual labels of data
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


def select_sleep_stage(awakenings, labels, info_df, stage = [4]):
    '''
    Function to select awakenings in specific sleep stages.
    
    :param awakenings (numpy array): data in shape (awakenings, \*whatever*\)
    :param labels (numpy array): label of each awakening
    :param info_df (pandas.dataframe): df containing information of the awakenings
    :param stage (list): optional, list integers from 1 to 3: NREM1, NREM2, NREM3. 4: REM
    :return: selected_aw as awakenings with the selected sleep stages
             selected_lab as labels of the selected awakenings
             selected_df as dataframe containing the informations of the selected awakenings
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


def create_images(input_filename, output_filename, image_size, normalize = False):
    """
    This function is used to generate images for all awakenings.
    It requires us to make sure that the data was saved in fold 'MANIPULATED_DATA_DIR'.
    
    :param input_filename: the name of input
    :param output_filename: the name of output
    :param image_size: the size of images that we want to generate.
    :param output: the results were saved as the 'output_filename' in MANIPULATED_DATA_DIR directory
    """

    #load data
    filename = MANIPULATED_DATA_DIR + input_filename    
    loaded = np.load(filename+'.npz')
    locs_3D = sio.loadmat(channels_coord)['locstemp']
    
    data = loaded['data']
    label = loaded['labels']
    locs_2D = map_3d_to_2d(locs_3D)
    
    print(f'Data loaded from {filename}.npz')
    print('Data shape:',data.shape)
    print('N. features for each electrode:', data.shape[2] // 256)

    #generate images for all awakenings:
    raw_images=[]
    for i in range(data.shape[0]):
        awakening = data[i]
        print(f'Processing awakening: {i+1}/{data.shape[0]}', end='\r')
        images = generate_images(locs_2D, awakening, image_size, normalize=normalize,
                   augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False,
                   verbose = 0)
        raw_images.append(images)
        
    raw_images = np.array(raw_images)    
    # Save results
    fileName =  MANIPULATED_DATA_DIR + \
                str(image_size)+'_'+str(image_size)+ '_' + output_filename
    np.savez_compressed(fileName, images = raw_images, labels = label )
    print(f'\nSaved in {fileName}.npz')
    
    
def create_videos(input_filename, video_size, slide):
    '''
    This function is used to generate videos from given images.
    
    :param video_size: number of frames in video
    :param slide: number of frames to slide in time
    :return: video, label
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
        Videos = generate_video_from_images(awakening, video_size, slide)
        videos_data.append(Videos)
    print ('\n')
    Videos_data = np.array(videos_data)
    return Videos_data, label


def create_and_save_videos(input_filename, output_filename, video_size, slide):
    '''
    This function is used to generate videos from images.
    
    :param input_filename:
    :param output_filename:
    :param video_size: number of frames in video
    :param slide: number of frames to slide in time 
    :param output: the results were saved as the output_filename
    '''

    label, Videos_data = create_videos(input_filename, video_size, slide)
    # Save results
    print('\nSaving... ')
    fileName =  MANIPULATED_DATA_DIR + \
                str(video_size)+'_'+str(slide)+ '_' + output_filename
    np.savez_compressed(fileName, videos = Videos_data, labels = label )
    print(f'Saved in {fileName}.npz')   
    

def readlabels(filename = 'Dream_reports_healtly.xlsx'):
    '''
    Read informations from the Excel file located in filename.
    It is the file provided by CHUV hospital with all the informations of the awakenings.

    :return: dataframe containing the informations.
    '''

    # Read trial ids and labels from excel 
    datalabels = pd.read_excel(open(filename,'rb'),\
                 dtype={'Subject_id':str, 'Quest_number':str, 'Stage':int, 'CE':float, 'Segment excluded':int})
    #drop the awaking recording without labels
    datalabels.dropna(subset=['CE'], inplace=True)
    datalabels['CE'] = datalabels['CE'].astype('int') 
    #drop the Segment excluded
    datalabels = datalabels[datalabels['Segment excluded'] == 0]
    datalabels.drop('Segment excluded', axis=1, inplace=True)
    datalabels.Quest_number = datalabels.Quest_number.apply(lambda x: 'S0'+x if len(x)<2 else 'S'+x)
    
    return datalabels


def prepare_raw_data(seconds=20, dest_dir='./'):
    """
    Read all raw data files, create matrix format and save. DATA_DIR set in config.py file 
    
    :param second: extract only last given seconds
    :param dest_dir: destination of the data
    """
    print('Reading Raw data')
    # Read the labels
    datalabels = readlabels(labels_file)
    # Read and save all data files path and with their names
    datafiles = [] 
    filenames = []
    missingfiles = []
    Sind = []
    
    for d in datadir:
        subjectpath = DATA_DIR_REG + d+ '/'
        trialfiles = os.listdir(subjectpath)
        for filename in trialfiles:
            datapath = subjectpath + filename
            #print(datapath)
            filenames.append(filename)
            datafiles.append(datapath)

    print('Total number of .mat files found: ',len(datafiles))
    
    all_data = []
    all_labels = []
    all_labels2 = []
    df_ind = []

    for rowid in datalabels.index:
        sid = datalabels.get_value(rowid, 'Subject_id')
        qn = datalabels.get_value(rowid, 'Quest_number')
        label = datalabels.get_value(rowid, 'CE')
        label2 = datalabels.get_value(rowid, 'Stage')
        
        # Find .mat file belongs to this trial
        fname = re.compile(r"^"+sid + "_.*_" + qn+ ".mat")
        fnamelist = list(filter(fname.search, filenames))
        # Check if any match
        if len(fnamelist)>0:
            print(sid, qn , fnamelist)
            ind = filenames.index(fnamelist[0])
            # Find the datapath of file
            fpath = datafiles[ind]
            df_ind.append(ind)
            # read the last 'second' seconds of the file and append it to the list
            arrays = {}
            try :
                f = h5py.File(fpath,'r')
                for k, v in f.items():
                    arrays[k] = np.array(v)

                mydata = (arrays['datavr'])[-(seconds*SAMPLING_RATE):,0:256]
                all_data.append(mydata)
                #all_labels.append(label)
                all_labels2.append(label2)   
                #print(fpath , 'DONE' )
                f.close()
                Sind.append(rowid)

            except Exception as e:
                print(e)
                missingfiles.append(fpath)


    if len(all_data) == len(datafiles):
        print('In raw, file reading is fine')
    else:
        print('In raw, some files are missed while reading')
        print(len(all_data) , len(datafiles))
        miss = np.setdiff1d(np.arange( len(datafiles)) , np.array( df_ind ))
        for m in miss:
            print('Missing files: ', datafiles[m])

    # save files into dest_dir
    print('Saving NPZ archive and info dataframes in:', dest_dir + str(seconds)+'_RAW_zip.npz')
    print('This may take some time ...')
    awakings = datalabels.ix[np.array(Sind)]  
    awakings.to_pickle(dest_dir +'awakenings_info_df_RAW.pkl')  
    np.savez_compressed(dest_dir + str(seconds)+'_RAW_zip.npz', data=all_data, labels=all_labels2)


def prepare_fft_sw_data(seconds=120, dest_dir='./'):
    """
    Read all fft data files, create matrix format and save. DATA_DIR set in config.py file 
    
    :param second: extract last given number of frames, not necessarily seconds. 
                   data in shape (seconds, 512) where each 256 column belongs to delta power and gamma power respectively
    :param dest_dir: destination of the data
    """
    print('Reading FFT with SW data')
    # Read the labels
    datalabels = readlabels(labels_file)
    # Read and save all data files path and with their names
    datafiles = [] 
    filenames = []
    missingfiles = []
    for d in datadirfftsw:
        subjectpath = DATA_DIR_FFT_SW + d+ '/'
        trialfiles = os.listdir(subjectpath)
        for filename in trialfiles:
            datapath = subjectpath + filename
            #print(datapath)
            filenames.append(filename)
            datafiles.append(datapath)

    print('Total number of .mat files found: ',len(datafiles))

    all_data = []
    all_labels = []
    all_labels2 = []
    df_ind = []
    Sind = []
    for rowid in datalabels.index:
        sid = datalabels.at[rowid,'Subject_id']
        qn = datalabels.at[rowid, 'Quest_number']
        label = datalabels.at[rowid, 'CE']
        label2 = datalabels.at[rowid, 'Stage']
        # Find .mat file belongs to this trial
        fname = re.compile(r"^"+sid + "_.*_" + qn+ "_DeltaGammaPSD.mat")
        fnamelist = list(set(filter(fname.search, filenames)))
        #print(sid, qn, fnamelist)
        
        # Check if any match
        if len(fnamelist)>0:
            ind = filenames.index(fnamelist[0])
            # Find the datapath of file
            fpath = datafiles[ind]
            df_ind.append(ind) 
            #print(f"subject {sid}, quest number {qn}, file names: {fnamelist}")
            #print(f"filenames index0 = {ind}, file path {fpath}")
            # read the last 30 second of the file and append it to the list
            try:
                a_trial = sio.loadmat(fpath)
                delta =  a_trial['delta'].T[-seconds:,0:256]
                gamma = a_trial['gamma'].T[-seconds:,0:256]
                windowed_trial = []
                for ind in range(0, delta.shape[0]):
                    concat = np.concatenate((delta[ind], gamma[ind]), axis=None)
                    windowed_trial.append(concat)  
                two_channel_data = np.asarray(windowed_trial)
                all_data.append(two_channel_data)
                all_labels.append(label)
                all_labels2.append(label2)
                Sind.append(rowid)
                print(f"Row id {rowid}, data shape:{two_channel_data.shape}, label1:{label}, label2:{label2}")
            except Exception as e:
                print(e)
                print('exception file: ', fpath)
                missingfiles.append(fpath)

    if len(all_data) == len(datafiles):
        print('FFT SW file reading is fine')
    else:
        print('At FFT SW, some files are missed while reading')
        print(len(all_data), len(datafiles))
        miss = np.setdiff1d(np.arange(len(datafiles)), np.array(df_ind))
        for m in miss:
            print('Missing files: ', datafiles[m])

    # save files into dest_dir
    print('Saving NPZ archive and info dataframes in:', dest_dir + str(seconds)+'_FFT_zip.npz')
    awakings = datalabels.loc[np.array(Sind)]
    awakings.to_pickle(dest_dir +'awakenings_info_df_FFT.pkl')
    np.savez_compressed(dest_dir + str(seconds)+'_FFT_zip.npz', data=all_data, labels=all_labels)
         
        
        
def main():
    DEST_DIR = MANIPULATED_DATA_DIR
    
    if len(sys.argv) < 3:
        print("Please give data type (raw|fft) and seconds|frame")
        sys.exit()
    
    try:
        if not os.path.exists(DEST_DIR):
            os.mkdir(DEST_DIR)
            print(f"Destination directory: {DEST_DIR}")
    except:
        print(f"Problem in creating destination directory: {DEST_DIR}")
        sys.exit()     
    
    try:
        data_type = sys.argv[1]
        seconds = int(sys.argv[2])
        print("Argument List:", str(sys.argv))
    except:
        print(f"Problem in casting parameteres data_type:{data_type}, seconds:{second}")
        
          
    if data_type == "raw": # Raw data acquisition
        print (f'Start reading raw data for {seconds} seconds.')    
        prepare_raw_data(seconds, dest_dir=DEST_DIR)
        print('Finished reading RAW data!')      
    elif data_type == "fft": # FFT with SW data acquisition
        print (f'Start reading fft data for {seconds} seconds.')    
        prepare_fft_sw_data(seconds, dest_dir=DEST_DIR)
        print('Finished reading FFT data!') 
    
    print(f'Data is ready to be used in: {DEST_DIR}')


    
if __name__ == '__main__':
      main()      
      
    
#if __name__ == '__main__':       
#    #generate images for raw data
#    create_video('32_32_FFT_SW_log_z-score','FFT_SW_videos_log_z-score', 50, 5)
        
        
        
        
        