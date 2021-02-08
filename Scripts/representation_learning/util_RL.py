import numpy as np
import pandas as pd

import torch 
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
1


def gen_data_for_representation_learning(images, n_predicted_frames,video_size, slide,N_after_videos=1):
    """
    generate videos from images for all awakenings.
    
    :param images: the image data with the shape (n_awaking, n_frames, n_channels, 32, 32)
    :param labels: the label of each awanking (n_awaking, )，DE. NDE
    :param video_size: the size of frames of video
    :param slide: the size of frames of sliding window
    :N_after_videos: the Number of images after the video as the predicted images, if you want the predicted image to 
                     be the one immediately after the video, set it as 1 (default).

        
    :return: the data of videos. 
    """
    n_awaking = images.shape[0]    
    num_video = (images.shape[1]-(video_size+N_after_videos))//slide + 1
    
    n_channels = images.shape[2]
    image_size = images.shape[3]
       
    x_tr_video = np.ones((n_awaking,num_video,video_size,n_channels,image_size,image_size))
    label_images = np.ones((n_awaking,num_video,n_predicted_frames*n_channels*image_size*image_size))

    for (idx, images_aw) in enumerate(images):
        for i in range(num_video):
                           
            a_video = images_aw[i*slide : i*slide+video_size]
            a_image = images_aw[i*slide+video_size + N_after_videos-1:i*slide+video_size+ N_after_videos-1+n_predicted_frames].flatten()
            
            x_tr_video[idx][i] = a_video
            label_images[idx][i] = a_image
    return np.array(x_tr_video), np.array(label_images)






def split_train_test_RL(data, label_images, label_classes, label_subjects, sleep_stage, infos_df, test_codes):
    """
    This function is used to split the train and test from the original data. 
    We choose the awaking in test_codes as our test set and the rest as the train set
    
    :param data: video data.[n_awaking, num_video,video_size, n_channels,32,32]
    :param label_images: the representation images.[n_awaking,num_video,n_channels,32,32 ]
    :param label_classes: the classes labels.[n_awaking]
    :param label_subjects: the subjects labels.[n_awaking]
    :param sleep_stage: the sleep stage, REM 1, NREM 0. [n_awaking]   
    :infos_df : the dataframe that contains each awaking recording information.
    :test_codes : the name of the awaing recordings.
    
    output:
    data_train_reshape: train dara
    data_test_reshape: test data
    labels_train: train labels
    labels_test: test labels
    """
    #find the test index
    test = infos_df[infos_df['name_code'].isin(test_codes)]
    test_index = test.index      
    #find the train index
    train = infos_df[~infos_df['name_code'].isin(test_codes)]
    train_index = train.index
    
    #check that the codes were correct
    if (len(test) != len(test_codes)) or (len(test) == 0 and len(test_codes) != 0):
        print(colored(f'Not able to find all the codes {test_codes}','red'))
        print('Exit.')
        sys.exit()
    
    #check that we don't lose data
    if (infos_df.index != (train_index.union(test_index))).any():
        print(colored(f'There are some data has been lost in the split data using codes: {test_codes}','red'))
        print('Exit.')
        sys.exit()        
    
    #split data to train and test accodirding to the index
    data_train = data[train_index]
    label_images_train = label_images[train_index]
    label_classes_train = label_classes[train_index]
    label_subjects_train = label_subjects[train_index]
    sleep_stage_train = sleep_stage[train_index]

    data_test = data[test_index]
    label_images_test = label_images[test_index]
    label_classes_test = label_classes[test_index]
    label_subjects_test = label_subjects[test_index]
    sleep_stage_test = sleep_stage[test_index]

    return data_train,label_images_train, label_classes_train,label_subjects_train,sleep_stage_train, data_test, label_images_test, label_classes_test, label_subjects_test,sleep_stage_test


def split_data_RL(x, y1,y2,y3,y4,ratio, seed=1):
    """
    Split the dataset based on the split ratio. If ratio is 0.8 you will have 
    80% of your data set dedicated to training and the rest dedicated to 
    testing. 
    
    this function will make sure there the ND labels are both in test/validationn set.
    y2 is label class.
    Return the training then testing sets (x_tr, x_te) and training then testing
    labels (y_tr, y_te).
    """
    
    NDE = np.where(y2 == 0)
    DE = np.where(y2 != 0)
    
    x_NDE = x[NDE]
    y1_NDE = y1[NDE]
    y2_NDE = y2[NDE]
    y3_NDE = y3[NDE]
    y4_NDE = y4[NDE]
    
    x_DE = x[DE]
    y1_DE = y1[DE]
    y2_DE = y2[DE]
    y3_DE = y3[DE]
    y4_DE = y4[DE]
    
    #Set seed
    np.random.seed(seed)
    x_NDErand = np.random.permutation(x_NDE)
    np.random.seed(seed)
    x_DErand = np.random.permutation(x_DE)    
    np.random.seed(seed)
    y1_NDErand = np.random.permutation(y1_NDE)
    np.random.seed(seed)
    y1_DErand = np.random.permutation(y1_DE)
    np.random.seed(seed)
    y2_NDErand = np.random.permutation(y2_NDE)
    np.random.seed(seed)
    y2_DErand = np.random.permutation(y2_DE)
    np.random.seed(seed)
    y3_NDErand = np.random.permutation(y3_NDE)
    np.random.seed(seed)
    y3_DErand = np.random.permutation(y3_DE)    
    np.random.seed(seed)
    y4_NDErand = np.random.permutation(y4_NDE)
    np.random.seed(seed)
    y4_DErand = np.random.permutation(y4_DE)  
    
    #Used to compute how many samples correspond to the desired ratio.
    limit_NDE = int(x_NDE.shape[0]*ratio)
    limit_DE = int(x_DE.shape[0]*ratio)
    
    x_NDE_tr = x_NDErand[:limit_NDE]
    x_NDE_te = x_NDErand[(limit_NDE):]
    x_DE_tr = x_DErand[:limit_DE]
    x_DE_te = x_DErand[(limit_DE):]
    x_tr = np.concatenate((x_NDE_tr,x_DE_tr),axis=0)
    x_te = np.concatenate((x_NDE_te,x_DE_te),axis=0)
    
    y1_NDE_tr = y1_NDErand[:limit_NDE]
    y1_NDE_te = y1_NDErand[(limit_NDE):]
    y1_DE_tr = y1_DErand[:limit_DE]
    y1_DE_te = y1_DErand[(limit_DE):]
    y1_tr = np.concatenate((y1_NDE_tr,y1_DE_tr),axis=0)
    y1_te = np.concatenate((y1_NDE_te,y1_DE_te),axis=0)
    
    y2_NDE_tr = y2_NDErand[:limit_NDE]
    y2_NDE_te = y2_NDErand[(limit_NDE):]
    y2_DE_tr = y2_DErand[:limit_DE]
    y2_DE_te = y2_DErand[(limit_DE):]
    y2_tr = np.concatenate((y2_NDE_tr,y2_DE_tr),axis=0)
    y2_te = np.concatenate((y2_NDE_te,y2_DE_te),axis=0)
    
    y3_NDE_tr = y3_NDErand[:limit_NDE]
    y3_NDE_te = y3_NDErand[(limit_NDE):]
    y3_DE_tr = y3_DErand[:limit_DE]
    y3_DE_te = y3_DErand[(limit_DE):]
    y3_tr = np.concatenate((y3_NDE_tr,y3_DE_tr),axis=0)
    y3_te = np.concatenate((y3_NDE_te,y3_DE_te),axis=0)
    
    y4_NDE_tr = y4_NDErand[:limit_NDE]
    y4_NDE_te = y4_NDErand[(limit_NDE):]
    y4_DE_tr = y4_DErand[:limit_DE]
    y4_DE_te = y4_DErand[(limit_DE):]
    y4_tr = np.concatenate((y4_NDE_tr,y4_DE_tr),axis=0)
    y4_te = np.concatenate((y4_NDE_te,y4_DE_te),axis=0)
    
    
    return x_tr, x_te, y1_tr, y1_te, y2_tr, y2_te, y3_tr, y3_te,y4_tr, y4_te





def reformat_data_labels_RL(data, label_images,label_classes,label_subjects,sleep_stage):
    """
    This function will reformate data as awaking * n_videos ad also at the same time expends labels 
    according to the data
    
    :param data: the data matrix [n_awanking, n_videos, n_channel,32,32]
    :param label_images: the representation images.[n_awaking,num_video,n_channels,32,32 ]
    :param label_classes: the classes labels.[n_awaking]
    :param label_subjects: the subejcts labels.[n_awaking]    
    :paem sleep_stage: the sleep stage, RAM 1, NORAM 0. [n_awaking]  

    output:
    data_reshaped:  [n_awanking*n_videos, n_channel,32,32]
    label_images_reshaped: [n_awanking*n_videos]
    label_classes_reshaped: [n_awanking*n_videos]
    sleep_stage_reshaped: [n_awanking*n_videos]  
    """

    # merge awakenings
    shp = data.shape
    data_reshaped = data.reshape(shp[0]*shp[1], shp[2], shp[3], shp[4], shp[5])
    shp = label_images.shape
    label_images_reshaped = label_images.reshape(shp[0]*shp[1], shp[2])

    # repeat labels
    n_rep = data.shape[1]
    label_classes_reshaped = np.repeat(label_classes, n_rep,axis=0)
    label_subjects_reshaped = np.repeat(label_subjects, n_rep,axis=0)
    sleep_stage_reshaped = np.repeat(sleep_stage, n_rep,axis=0)
    
    
    return data_reshaped, label_images_reshaped, label_classes_reshaped, label_subjects_reshaped, sleep_stage_reshaped


def dataloader_pytorch_RL(x, y, mini_batch_size,shuffle=True):
    """
    This function take data x and label y to convert them as a dataloader.
    if we only want to convert x, we could set y as np.empty(0).
    
    """
    
    #convert to numpy
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    #reshape to [n_samples,n_channels,n_frames,32,32]  
    x = x.permute(0,2,1,3,4)
    
    #change the type
    x = x.type(torch.float)
    y = y.type(torch.float)
    
    if len(y) != 0:
        #Dataloader
        dataset = TensorDataset(x, y)
        dataloader = DataLoader(dataset, batch_size= mini_batch_size,shuffle=True)   
    else:
        dataloader = DataLoader(x, batch_size= mini_batch_size,shuffle=False)   

    return dataloader


def count_parameters(model):
    """
    this is the function is used to comput how many parameters that the model has.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
