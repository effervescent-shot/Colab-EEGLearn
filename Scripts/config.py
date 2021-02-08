# Directory with the original data. Path must be given relative to te position of the Scripts folder


##############################################################
# NOTE:
# The directory with the data should also contain a folder called
# 'dataset_info' with the files channelcoords.mat and 'Dream_reports_healthy.xlsx'
# Refer to the folder ../fake_dream_data for an example of the structure of DATA_DIR with randomly generated data.


#FAKE_DATA_DIR = '../fake_dream_data/' # use this line to test the scripts with fake data
DATA_DIR = '../../dream_data_original/'


##############################################################
# Directory where to save the 'manipulated' data in .npz format

FAKE_MANIPULATED_DATA_DIR = '../fake_dream_data_manipulated/'
MANIPULATED_DATA_DIR = '../../dream_data_manipulated/'
