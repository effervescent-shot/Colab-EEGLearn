################################################################################
# Authors: Nihal Ezgi Yuceturk, Hongyu Luo and Nicola Ischia       (EPFL)      #
#                                                                              #
# Date:                                                                        #
#                                                                              #
# Data acquisition routines                                                    #
################################################################################

import warnings
warnings.filterwarnings("ignore")

import h5py
import os, re, sys
import numpy as np
import pandas as pd
import scipy.io as sio

from config import DATA_DIR, MANIPULATED_DATA_DIR
from helpers import ask_user_0_1, ask_user_integer

# Generate proper directories
DATA_DIR_REG = DATA_DIR + 'dream_data/'
DATA_DIR_FFT_SW = DATA_DIR + 'dream_data_fft_SW/'

channels_coord = DATA_DIR + 'dataset_info/channelcoords.mat'

# datadir = os.listdir(DATA_DIR_REG)
datadirfftsw = os.listdir(DATA_DIR_FFT_SW)

SAMPLING_RATE = 500


def readlabels(filename = 'Dream_reports_healtly.xlsx'):
    '''
    Read informations from the Excel file located in filename.
    It is the file provided by CHUV hospital with all the informations of the 
    awakenings.

    Return a dataframe containing the informations.
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


def prepare_raw_data(datalabels, second = 20, dest_dir = './'):
    """
    read all data files, create matrix format and save.
    NOTE: it requires that DATA_DIR in config.py file have been properly set.
    second: extract only last given seconds
    dest_dir: destination of the data
    """
    print('Reading Raw data')
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

                mydata = (arrays['datavr'])[-(second*SAMPLING_RATE):,0:256]
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
    print('Saving NPZ archive and info dataframes in:', dest_dir)
    print('This may take some time ...')
    awakings = datalabels.ix[np.array(Sind)]  
    awakings.to_pickle(dest_dir +'awakenings_info_df_raw_data.pkl')  
    fileName = str(second)+'sec_raw_data_zip'
    np.savez_compressed(dest_dir + fileName, data=all_data, labels=all_labels2)
    
        

def prepare_fft_sw_data(datalabels, dest_dir = './'):
    """
    read all data files, create matrix format and save
    matrix format for each trial is 60 * 512 where first 256 column belongs to delta power and last 256 belong to gamma power
    dest_dir: destination of the data
    """
    print('Reading FFT with SW data')
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
        sid = datalabels.get_value(rowid, 'Subject_id')
        qn = datalabels.get_value(rowid, 'Quest_number')
        label = datalabels.get_value(rowid, 'CE')
        label2 = datalabels.get_value(rowid, 'Stage')
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
                delta =  a_trial['delta'].T[:,0:256]
                gamma = a_trial['gamma'].T[:,0:256]
                windowed_trial = []
                for ind in range(0, delta.shape[0]):
                    concat = np.concatenate((delta[ind], gamma[ind]), axis=None)
                    windowed_trial.append(concat)  
                two_channel_data = np.asarray(windowed_trial)
                all_data.append(two_channel_data)
                all_labels.append(label)
                all_labels2.append(label2)
                Sind.append(rowid)
                #print(f"Row id {rowid}, data shape:{two_channel_data.shape}, label1:{label}, label2:{label2}")
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
    print('Saving NPZ archive and info dataframes in:', dest_dir)
    # print('This may take some time ...')
    awakings = datalabels.ix[np.array(Sind)]
    awakings.to_pickle(dest_dir +'awakenings_info_df_FFT_SW.pkl')
    np.savez_compressed(dest_dir + 'FFT_data_SW_zip.npz', data=all_data, labels=all_labels)
   
               
        

def main():
    datalabels = readlabels(DATA_DIR + 'dataset_info/Dream_reports_healthy.xlsx')
    DEST_DIR = MANIPULATED_DATA_DIR
    try:
        os.mkdir(DEST_DIR)
    except:
        pass

    # Raw data acquisition
    user_input = ask_user_0_1('Do you want to acquire raw data? (0=no, 1=yes) ')
    if user_input:
        seconds = ask_user_integer('How many (last) seconds do you want ' + 
                                   'to acquire? (integer value) ')
        print ('Start reading raw data for ', seconds, ' seconds.')    
        prepare_raw_data(datalabels, second=seconds, dest_dir=DEST_DIR)
        print('Finished reading raw data!') 
    
    # FFT with SW data acquisition
    user_input = ask_user_0_1('\n\nDo you want to acquire FFT with SW data? (0=no, 1=yes) ',
                              'Start reading FFT with SW data.')
    if user_input == True:
        prepare_fft_sw_data(datalabels, dest_dir=DEST_DIR)
        print('Finished reading FFT with SW data!') 
    print(f'Data is ready to be used in: {DEST_DIR}')



# Standard boilerplate to call the main() function to begin
# the program.
if __name__ == '__main__':
      main()      
        
        
        
        
    