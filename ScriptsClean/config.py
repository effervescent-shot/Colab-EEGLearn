import os
import sys

# Data directories
FAKE_DATA_DIR = '../fake_dream_data/' 
DATA_DIR = '../../dream_data_original/'
channels_coord = DATA_DIR + 'dataset_info/channelcoords.mat'
labels_file= DATA_DIR + 'dataset_info/Dream_reports_healthy.xlsx'

FAKE_MANIPULATED_DATA_DIR = '../fake_dream_data_manipulated/'
MANIPULATED_DATA_DIR = '../../dream_data_manipulated/'

# Generate proper directories
DATA_DIR_REG = DATA_DIR + 'dream_data/'
DATA_DIR_FFT_SW = DATA_DIR + 'dream_data_fft_SW/'
# datadir = os.listdir(DATA_DIR_REG)
#datadirfftsw = os.listdir(DATA_DIR_FFT_SW)

SAMPLING_RATE = 500
