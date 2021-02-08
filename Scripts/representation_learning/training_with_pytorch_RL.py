import os, sys, ast
import math as m
import numpy as np
import pickle
import pandas as pd
import scipy.io as sio


# pytorch stuff
from util_RL import *
from pytorch_models_RL import *
from training_with_pytorch_RL import *
from utils_pytorch_RL import *


def train_for_n_epochs_RL(model, loader_train, loader_test, current_train, L2_regularizer, L1_regularizer,
                        learning_rate,device, n_epoch,patient,output_dir = './'):
    Train_loss = []
    Test_loss = []

    stop_count = 0
    best_test_loss = 0

    for k in range(n_epoch):
        train_loss = train_model_RL(model, loader_train,L2_regularizer,learning_rate,L1_regularizer,device)
        test_loss = compute_nb_errors_RL(model, loader_test,L2_regularizer,learning_rate,L1_regularizer,device)
                                    
                              
        print('Epoch {}/{} train_loss {}  test_loss {} '.format(k,n_epoch,round(train_loss,2), round(test_loss,2)))

        # early_stoping
        if k ==0:
            best_test_loss = test_loss
        else:
            if test_loss < best_test_loss: 
                stop_count = 0
                best_test_loss = test_loss
            else:
                stop_count += 1

                if stop_count >= patient: # stop training if val_acc dose not imporve for over patient
                    break       
                                        
        Train_loss.append(train_loss)
        Test_loss.append(test_loss)
    
    print(f'the best test loss is : {best_test_loss}')
    checkpoint_name = output_dir + f'RL_best_model_gridsearch{current_train}'
    #torch.save(model.state_dict(), checkpoint_name)
    torch.save(model, checkpoint_name)
    print(f'Checkpoint saved in {checkpoint_name}')
    
    return np.asarray(Train_loss), np.asarray(Test_loss),best_test_loss



def representation_learning_training(loader_train, loader_val,
                                loader_train_encoder,loader_val_encoder, loader_test_encoder,
                                label_classes_train,label_classes_val, label_classes_test,
                                label_subjects_train,label_subjects_val, label_subjects_test,
                                sleep_stage_train,sleep_stage_val, sleep_stage_test,
                                n_epoch, model_type = 'convpool_conv1d', GPU=0,
                n_channels=2, n_frames = 10, n_predicted_frames=1, patient=10,output_dir = '.'):

    # Using GPU
    if GPU != -1:
        gpu_string = f'cuda:{GPU}'
        device = torch.device(gpu_string if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # For each model we have different sets of parameters to tune
    if model_type == 'convpool_conv1d':

        # Hyperparameters to tune
        kernels = [3]
        d_layers = [128]
        #the kernel size on the time dimension in id conv
        dropoutRates = [0.3]
        L2_regularizers = [1e-5]
        k_size_1ds = [3]
        L1_regularizers = [0]

        # We don't tune these parameters
        act_fn = nn.ReLU
        learning_rate = 5e-6

        #save result
        resultsdf = pd.DataFrame(columns=['kernel','d_layer','dropoutRate','act_fn','L2_regularizer','L1_regularizer',
                                       'learning_rate', 'k_size_1d', 'n_parameters','Train_loss','Test_loss','best_test_loss'])
        current_train = 0
        total_parameters = len(kernels)*len(d_layers)*len(dropoutRates)*len(L2_regularizers)*len(L1_regularizers)*len(k_size_1ds)

        for kernel in kernels:
            for d_layer in d_layers:
                for dropoutRate in dropoutRates:
                    for L2_regularizer in L2_regularizers:
                        for L1_regularizer in L1_regularizers:
                            for k_size_1d in k_size_1ds:
                                current_train = current_train + 1
                                print('\n--------------------------------------')
                                print('\tSTART TRAIN n. '+
                                    f'{current_train}/{total_parameters}')
                                print('kernel {}, d_layer {}, dropoutRate {}, act_fn {}, L2_regularizer {}, learning_rate {}'.format(
                                    kernel, d_layer, dropoutRate, act_fn, L2_regularizer, learning_rate
                                    )
                                        )
                                print('---------------------------------------\n')

                                model = RL_convpool_conv1d(act_fn=act_fn, kernel=kernel,
                                            d_layer=d_layer,
                                            n_channels=n_channels, 
                                            dropoutRate=dropoutRate,
                                            n_frames=n_frames,
                                            k_size_1d=k_size_1d,n_predicted_frames=n_predicted_frames)
                                model = model.to(device)

                                # compute the number of parameters in the model
                                n_parameters = count_parameters(model)
                                print('number of paramaters {}'.format(n_parameters))
                                
                                Train_loss,Test_loss,best_test_loss = train_for_n_epochs_RL(model,
                                            loader_train, loader_val, current_train,L2_regularizer, L1_regularizer,
                                            learning_rate, device, n_epoch,patient, output_dir = output_dir)
                                
                                resultsdf.loc[current_train-1] = [kernel,d_layer,dropoutRate,act_fn,L2_regularizer,L1_regularizer,
                                        learning_rate, k_size_1d, n_parameters,Train_loss, Test_loss,best_test_loss]

                                # generate the encoder
                                # get the x_train_encoder, x_test_encoder (numpy array)
                                print('start to generate the encoder for train')
                                x_train_encoder = get_encoder(loader_train_encoder,model,GPU)
                                print('start to generate the encoder for validation')
                                x_val_encoder = get_encoder(loader_val_encoder,model,GPU)
                                print('start to generate the encoder for test')
                                x_test_encoder = get_encoder(loader_test_encoder,model,GPU)

                                # save the encoder data
                                print('saving the encoder data')
                                df_filename = output_dir + f'encoder_data_gridsearch{current_train}'
                                np.savez_compressed(df_filename, x_train_encoder = x_train_encoder, x_val_encoder = x_val_encoder, x_test_encoder = x_test_encoder,
                                label_classes_train = label_classes_train, label_classes_val = label_classes_val, label_classes_test = label_classes_test,
                                label_subjects_train = label_subjects_train, label_subjects_val = label_subjects_val, label_subjects_test = label_subjects_test,
                                sleep_stage_train = sleep_stage_train,sleep_stage_val = sleep_stage_val,
                                sleep_stage_test=sleep_stage_test)
                                print(f'encoder data saved in: {df_filename}' )

    
    elif model_type == '2Dconv_max':

        # Hyperparameters to tune
        kernels = [3]
        d_layers = [128]
        dropoutRates = [0.3]
        L2_regularizers = [1e-5]
        L1_regularizers = [1e-5]

        time_pool_sizes = [10] 
        # fixed
        act_fn=nn.ReLU
        learning_rate = 0.000005

        #save result
        L1_regularizers = [1e-5]
        resultsdf = pd.DataFrame(columns=['kernel','d_layer','dropoutRate','act_fn','L2_regularizer','L1_regularizer',
                                       'learning_rate', 'time_pool_size','n_parameters', 'Train_loss','Test_loss','best_test_loss'])
        current_train = 0
        total_parameters = len(kernels)*len(d_layers)*len(dropoutRates)*len(L2_regularizers)*len(L1_regularizers)*len(time_pool_sizes)

        for kernel in kernels:
            for d_layer in d_layers:
                for dropoutRate in dropoutRates:
                    for L2_regularizer in L2_regularizers:
                        for L1_regularizer in L1_regularizers:
                            for time_pool_size in time_pool_sizes:
                                current_train = current_train + 1
                                print('\n--------------------------------------')
                                print('\tSTART TRAIN n. '+
                                    f'{current_train}/{total_parameters}')
                                print('---------------------------------------\n')

                                model = RL_2DConv(act_fn=act_fn, kernel=kernel,
                                            d_layer=d_layer,
                                            n_channels=n_channels,
                                            dropoutRate=dropoutRate,
                                            n_frames=n_frames,
                                            time_pool_size=time_pool_size,n_predicted_frames=n_predicted_frames)

                                model = model.to(device)

                                # compute the number of parameters in the model
                                n_parameters = count_parameters(model)
                                print('number of paramaters {}'.format(n_parameters))

                                Train_loss,Test_loss, best_test_loss = train_for_n_epochs_RL(model,
                                            loader_train, loader_val, current_train, L2_regularizer, L1_regularizer,
                                            learning_rate, device, n_epoch,patient, output_dir = output_dir)
                                

                                resultsdf.loc[current_train-1] = [kernel,d_layer,dropoutRate,act_fn,L2_regularizer, L1_regularizer,
                                        learning_rate, time_pool_size, n_parameters, Train_loss, Test_loss, best_test_loss]

                                # generate the encoder
                                # get the x_train_encoder, x_test_encoder (numpy array)
                                print('start to generate the encoder for train')
                                x_train_encoder = get_encoder(loader_train_encoder,model,GPU)
                                print('start to generate the encoder for validation')
                                x_val_encoder = get_encoder(loader_val_encoder,model,GPU)
                                print('start to generate the encoder for test')
                                x_test_encoder = get_encoder(loader_test_encoder,model,GPU)

                               # save the encoder data
                                print('saving the encoder data')
                                df_filename = output_dir + f'encoder_data_gridsearch{current_train}'
                                np.savez_compressed(df_filename, x_train_encoder = x_train_encoder, x_val_encoder = x_val_encoder, x_test_encoder = x_test_encoder,
                                label_classes_train = label_classes_train, label_classes_val = label_classes_val, label_classes_test = label_classes_test,
                                label_subjects_train = label_subjects_train, label_subjects_val = label_subjects_val, label_subjects_test = label_subjects_test,
                                sleep_stage_train = sleep_stage_train,sleep_stage_val = sleep_stage_val,
                                sleep_stage_test=sleep_stage_test)
                                print(f'encoder data saved in: {df_filename}' )



    elif model_type == 'mix':

        # Hyperparameters to tune
        kernels = [3]
        d_layers = [128]
        dropoutRates = [0.3]
        L2_regularizers = [1e-5]
        L1_regularizers = [1e-5]

        #the kernel size on the time dimension in id conv
        k_size_1ds = [3]
        lstm_units_list = [128]
        n_lstm_layers_list = [1]
        # Fixed
        act_fn=nn.ReLU
        learning_rate =  0.000005

        #save result
        resultsdf = pd.DataFrame(columns=['kernel','d_layer','dropoutRate','act_fn','L2_regularizer','L1_regularizer',
                                       'learning_rate', 'k_size_1d', 
                                       'lstm_units', 'n_lstm_layers', 'n_parameters',
                                       'Train_loss','Test_loss','best_test_loss'])

        current_train = 0
        total_parameters = len(kernels)*len(d_layers)*len(dropoutRates)*len(L2_regularizers)*len(L1_regularizers)*\
                           len(k_size_1ds)*len(lstm_units_list)*len(n_lstm_layers_list)

        for kernel in kernels:
            for d_layer in d_layers:
                for dropoutRate in dropoutRates:
                    for L2_regularizer in L2_regularizers:
                        for L1_regularizer in L1_regularizers:
                            for k_size_1d in k_size_1ds:
                                for lstm_units in lstm_units_list:
                                    for n_lstm_layers in n_lstm_layers_list:
                                        current_train = current_train + 1
                                        print('\n--------------------------------------')
                                        print('\tSTART TRAIN n. '+
                                            f'{current_train}/{total_parameters}')
                                        print('---------------------------------------\n')

                                        model = RL_Mix(act_fn=act_fn, kernel=kernel,
                                                d_layer=d_layer,
                                                n_channels=n_channels, 
                                                dropoutRate=dropoutRate,
                                                n_frames=n_frames,
                                                k_size_1d=k_size_1d,num_units=lstm_units, 
                                                n_LSTM_layers = n_lstm_layers,n_predicted_frames=n_predicted_frames)

                                        model = model.to(device)

                                        # compute the number of parameters in the model
                                        n_parameters = count_parameters(model)
                                        print('number of paramaters {}'.format(n_parameters))

                                        Train_loss,Test_loss,best_test_loss = train_for_n_epochs_RL(model,
                                            loader_train, loader_val, current_train, L2_regularizer, L1_regularizer,
                                            learning_rate, device, n_epoch,patient, output_dir = output_dir)
                                
                                        resultsdf.loc[current_train-1] = [kernel,d_layer,dropoutRate,act_fn,L2_regularizer,L1_regularizer,
                                                               learning_rate, k_size_1d, 
                                                               lstm_units,n_lstm_layers, n_parameters,
                                                               Train_loss,Test_loss,best_test_loss]


                                        # generate the encoder
                                        # get the x_train_encoder, x_test_encoder (numpy array)
                                        print('start to generate the encoder for train')
                                        x_train_encoder = get_encoder(loader_train_encoder,model,GPU)
                                        print('start to generate the encoder for validation')
                                        x_val_encoder = get_encoder(loader_val_encoder,model,GPU)
                                        print('start to generate the encoder for test')
                                        x_test_encoder = get_encoder(loader_test_encoder,model,GPU)

                                        # save the encoder data
                                        print('saving the encoder data')
                                        df_filename = output_dir + f'encoder_data_gridsearch{current_train}'
                                        np.savez_compressed(df_filename, x_train_encoder = x_train_encoder, x_val_encoder = x_val_encoder, x_test_encoder = x_test_encoder,
                                        label_classes_train = label_classes_train, label_classes_val = label_classes_val, label_classes_test = label_classes_test,
                                        label_subjects_train = label_subjects_train, label_subjects_val = label_subjects_val, label_subjects_test = label_subjects_test,
                                        sleep_stage_train = sleep_stage_train,sleep_stage_val = sleep_stage_val,
                                        sleep_stage_test=sleep_stage_test)
                                        print(f'encoder data saved in: {df_filename}' )


    elif model_type == 'LSTM':

        # Hyperparameters to tune
        kernels = [3]
        d_layers = [128] #512 128
        dropoutRates = [0.3]
        L2_regularizers = [1e-5]
        L1_regularizers = [0]

        lstm_units_list = [128]
        n_lstm_layers_list = [1]
        # Fixed
        act_fn=nn.ReLU
        learning_rate =  0.000005

        #save result
        resultsdf = pd.DataFrame(columns=['kernel','d_layer','dropoutRate','act_fn','L2_regularizer','L1_regularizer',
                                       'learning_rate', 
                                       'lstm_units', 'n_lstm_layers', 'n_parameters',
                                       'Train_loss','Test_loss','best_test_loss'])
        current_train = 0
        total_parameters = len(kernels)*len(d_layers)*len(dropoutRates)*len(L2_regularizers)*len(L1_regularizers)*\
                           len(lstm_units_list)*len(n_lstm_layers_list)

        for kernel in kernels:
            for d_layer in d_layers:
                for dropoutRate in dropoutRates:
                    for L2_regularizer in L2_regularizers:
                        for L1_regularizer in L1_regularizers:
                            for lstm_units in lstm_units_list:
                                for n_lstm_layers in n_lstm_layers_list:
                                    current_train = current_train + 1
                                    print('\n--------------------------------------')
                                    print('\tSTART TRAIN n. '+
                                        f'{current_train}/{total_parameters}')
                                    print('---------------------------------------\n')

                                    model = RL_LSTM(act_fn=act_fn, kernel=kernel,
                                            d_layer=d_layer,
                                            n_channels=n_channels, 
                                            dropoutRate=dropoutRate,
                                            n_frames=n_frames,
                                            num_units=lstm_units, n_LSTM_layers = n_lstm_layers,n_predicted_frames=n_predicted_frames)

                                    model = model.to(device)

                                    # compute the number of parameters in the model
                                    n_parameters = count_parameters(model)
                                    print('number of paramaters {}'.format(n_parameters))

                                    Train_loss,Test_loss,best_test_loss = train_for_n_epochs_RL(model,
                                            loader_train, loader_val,current_train, L2_regularizer, L1_regularizer,
                                            learning_rate, device, n_epoch,patient, output_dir = output_dir)
                                
                                    resultsdf.loc[current_train-1] = [kernel,d_layer,dropoutRate,act_fn,L2_regularizer,L1_regularizer,
                                            learning_rate,
                                            lstm_units, n_lstm_layers, n_parameters,
                                            Train_loss, Test_loss, best_test_loss]    

                                    # generate the encoder
                                    # get the x_train_encoder, x_test_encoder (numpy array)
                                    print('start to generate the encoder for train')
                                    x_train_encoder = get_encoder(loader_train_encoder,model,GPU)
                                    print('start to generate the encoder for validation')
                                    x_val_encoder = get_encoder(loader_val_encoder,model,GPU)
                                    print('start to generate the encoder for test')
                                    x_test_encoder = get_encoder(loader_test_encoder,model,GPU)

                                    # save the encoder data
                                    print('saving the encoder data')
                                    df_filename = output_dir + f'encoder_data_gridsearch{current_train}'
                                    np.savez_compressed(df_filename, x_train_encoder = x_train_encoder, x_val_encoder = x_val_encoder, x_test_encoder = x_test_encoder,
                                    label_classes_train = label_classes_train, label_classes_val = label_classes_val, label_classes_test = label_classes_test,
                                    label_subjects_train = label_subjects_train, label_subjects_val = label_subjects_val, label_subjects_test = label_subjects_test,
                                    sleep_stage_train = sleep_stage_train,sleep_stage_val = sleep_stage_val,
                                    sleep_stage_test=sleep_stage_test)
                                    print(f'encoder data saved in: {df_filename}' )


    else:
        raise ValueError("Model not supported []")

    return resultsdf

