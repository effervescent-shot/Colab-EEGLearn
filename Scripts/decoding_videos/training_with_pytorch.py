import os, sys, ast
import math as m
import numpy as np
import pickle
import pandas as pd
import scipy.io as sio
from termcolor import colored
import time

from config import *
from pytorch_models import *


from utils_pytorch import *
print(colored('Pytorch utils without SWA','cyan'))

# from utils_pytorch_withSWA import *
# print(colored('Pytorch utils with SWA','yellow'))



def train_for_n_epochs(model, loader_train, loader_val, L2_regularizer, L1_regularizer,
                        learning_rate, device, n_epoch, loss_weights = [1,1],
                        save_weights = False, output_dir = '.'):
    Train_loss = []
    Train_accuracy = []
    Val_loss = []
    Val_accuracy = []
    for k in range(n_epoch):
        start = time.time()
        train_loss, train_accuracy = train_model(model, loader_train, 
                                    L2_regularizer,learning_rate,L1_regularizer,
                                    device, loss_weights)
        val_loss, val_accuracy = compute_nb_errors(model, loader_val,
                                    L2_regularizer,learning_rate,L1_regularizer,
                                    device, loss_weights)
        elapsed_time = time.time() - start                            


        val_acc_to_print = [round(i,2) for i in val_accuracy]
        tr_acc_to_print = [round(i,2) for i in train_accuracy]
        print_str = f'Epoch {k+1}\{n_epoch} --> elapsed time {round(elapsed_time,2)}s;'\
                    f' loss weights{loss_weights}s; train_loss {round(train_loss,2)};'\
                    f' val_loss {round(val_loss,2)};\n' \
                    f'Standard acc, dream acc, no dream acc, balanced acc:'\
                    f' train ->  {tr_acc_to_print},    validation ->{val_acc_to_print}'
        print(print_str)

        Train_loss.append(train_loss)
        Train_accuracy.append(train_accuracy)
        Val_loss.append(val_loss)
        Val_accuracy.append(val_accuracy) 
        if save_weights:
            if (k%5 == 0):
                checkpoint_name = output_dir + f'epoch_{k}'
                torch.save(model, checkpoint_name)
                print(f'Checkpoint saved in {checkpoint_name}')

    return np.asarray(Train_loss), np.asarray(Train_accuracy), np.asarray(Val_loss), np.asarray(Val_accuracy)

    


def grid_search_torch(loader_train, loader_val, n_epoch, 
                      model_type = 'convpool_conv1d', GPU=-1, n_channels=2, 
                      n_classes=2, n_frames = 10, loss_weights = [1,1], 
                      save_weights = False, output_dir = '.'):

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
        learning_rate =  5e-06

        #save result
        resultsdf = pd.DataFrame(columns=['kernel','d_layer','dropoutRate','act_fn','L2_regularizer','L1_regularizer',
                                       'learning_rate', 'k_size_1d', 'n_parameters','Train_loss','Train_accuracy','Test_loss','Test_accuracy'])
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

                                model = Net_base_convpool_conv1d(act_fn=act_fn, kernel=kernel,
                                            d_layer=d_layer,
                                            n_channels=n_channels, n_classes=n_classes,
                                            dropoutRate=dropoutRate,
                                            n_frames=n_frames,
                                            k_size_1d=k_size_1d)
                                model = model.to(device)

                                # compute the number of parameters in the model
                                n_parameters = count_parameters(model)
                                print('number of paramaters {}'.format(n_parameters))

                                
                                Train_loss, Train_accuracy, \
                                Val_loss, Val_accuracy = train_for_n_epochs(model,
                                            loader_train, loader_val, L2_regularizer, L1_regularizer,
                                            learning_rate, device, n_epoch, loss_weights,
                                            save_weights = save_weights, output_dir = output_dir)
                                

                                resultsdf.loc[current_train-1] = [kernel,d_layer,dropoutRate,act_fn,L2_regularizer,L1_regularizer,
                                        learning_rate, k_size_1d, n_parameters,Train_loss, Train_accuracy, Val_loss, Val_accuracy]
    
    
    elif model_type == '2Dconv_max':

        # Hyperparameters to tune
        kernels = [3]
        d_layers = [128]
        dropoutRates = [0.3]
        L2_regularizers = [1e-5]
        L1_regularizers = [0]

        time_pool_sizes = [10] 
        # fixed
        act_fn=nn.ReLU
        learning_rate = 5e-06

        #save result
        resultsdf = pd.DataFrame(columns=['kernel','d_layer','dropoutRate','act_fn','L2_regularizer','L1_regularizer',
                                       'learning_rate', 'time_pool_size','n_parameters', 'Train_loss','Train_accuracy','Test_loss','Test_accuracy'])
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

                                model = Net_base_2DConv(act_fn=act_fn, kernel=kernel,
                                            d_layer=d_layer,
                                            n_channels=n_channels, n_classes=n_classes,
                                            dropoutRate=dropoutRate,
                                            n_frames=n_frames,
                                            time_pool_size=time_pool_size)

                                model = model.to(device)

                                # compute the number of parameters in the model
                                n_parameters = count_parameters(model)
                                print('number of paramaters {}'.format(n_parameters))

                                Train_loss, Train_accuracy, \
                                Val_loss, Val_accuracy = train_for_n_epochs(model,
                                            loader_train, loader_val, L2_regularizer, L1_regularizer,
                                            learning_rate, device, n_epoch, loss_weights,
                                            save_weights = save_weights, output_dir = output_dir)
                                

                                resultsdf.loc[current_train-1] = [kernel,d_layer,dropoutRate,act_fn,L2_regularizer, L1_regularizer,
                                        learning_rate, time_pool_size, n_parameters, Train_loss, Train_accuracy, Val_loss, Val_accuracy]

    elif model_type == 'mix':

        # Hyperparameters to tune
        kernels = [3]
        d_layers = [128]
        dropoutRates = [0.5]
        L2_regularizers = [5e-5]
        L1_regularizers = [0]

        #the kernel size on the time dimension in id conv
        k_size_1ds = [3]
        lstm_units_list = [256]
        n_lstm_layers_list = [1]
        # Fixed
        act_fn=nn.ReLU
        learning_rate = 5e-06

        #save result
        resultsdf = pd.DataFrame(columns=['kernel','d_layer','dropoutRate','act_fn','L2_regularizer','L1_regularizer',
                                       'learning_rate', 'k_size_1d', 
                                       'lstm_units', 'n_lstm_layers', 'n_parameters',
                                       'Train_loss','Train_accuracy','Test_loss','Test_accuracy'])
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

                                        model = Net_base_Mix(act_fn=act_fn, kernel=kernel,
                                                d_layer=d_layer,
                                                n_channels=n_channels, n_classes=n_classes,
                                                dropoutRate=dropoutRate,
                                                n_frames=n_frames,
                                                k_size_1d=k_size_1d,num_units=lstm_units, 
                                                n_LSTM_layers = n_lstm_layers)

                                        model = model.to(device)

                                        # compute the number of parameters in the model
                                        n_parameters = count_parameters(model)
                                        print('number of paramaters {}'.format(n_parameters))

                                        Train_loss, Train_accuracy, \
                                        Val_loss, Val_accuracy = train_for_n_epochs(model,
                                                loader_train, loader_val, L2_regularizer, L1_regularizer,
                                                learning_rate, device, n_epoch, loss_weights,
                                                save_weights = save_weights, output_dir = output_dir)
                                
                                        resultsdf.loc[current_train-1] = [kernel,d_layer,
                                            dropoutRate,act_fn,L2_regularizer,L1_regularizer,
                                            learning_rate, k_size_1d,
                                            lstm_units, n_lstm_layers, n_parameters,
                                            Train_loss, Train_accuracy, Val_loss, Val_accuracy]

    elif model_type == 'LSTM_reduced':

        # Hyperparameters to tune
        kernels = [3]
        d_layers = [256]
        dropoutRates = [0.2,0.35,0.5]
        L2_regularizers = [1e-6,5e-7]
        L1_regularizers = [0]

        lstm_units_list = [512]
        n_lstm_layers_list = [1]
        # Fixed
        act_fn=nn.ReLU
        learning_rates = [1e-6,5e-6] #7.5e-06

        #save result
        resultsdf = pd.DataFrame(columns=['kernel','d_layer','dropoutRate','act_fn','L2_regularizer','L1_regularizer',
                                       'learning_rate', 
                                       'lstm_units', 'n_lstm_layers', 'n_parameters',
                                       'Train_loss','Train_accuracy','Test_loss','Test_accuracy'])
        current_train = 0
        total_parameters = len(kernels)*len(d_layers)*len(dropoutRates)*len(L2_regularizers)*len(L1_regularizers)*\
                           len(lstm_units_list)*len(n_lstm_layers_list)*len(learning_rates)

        for kernel in kernels:
            for d_layer in d_layers:
                for dropoutRate in dropoutRates:
                    for L2_regularizer in L2_regularizers:
                        for L1_regularizer in L1_regularizers:
                            for lstm_units in lstm_units_list:
                                for n_lstm_layers in n_lstm_layers_list:
                                    for learning_rate in learning_rates:
                                        current_train = current_train + 1
                                        print('\n--------------------------------------')
                                        print('\tSTART TRAIN n. '+
                                            f'{current_train}/{total_parameters}')
                                        print('---------------------------------------\n')

                                        model = LSTM_with_reduced_CNN(act_fn=act_fn,
                                                kernel=kernel,
                                                d_layer=d_layer,
                                                n_channels=n_channels, n_classes=n_classes,
                                                dropoutRate=dropoutRate,
                                                n_frames=n_frames,
                                                num_units=lstm_units, n_LSTM_layers = n_lstm_layers)

                                        model = model.to(device)

                                        # compute the number of parameters in the model
                                        n_parameters = count_parameters(model)
                                        print('number of paramaters {}'.format(n_parameters))

                                        Train_loss, Train_accuracy, \
                                        Val_loss, Val_accuracy = train_for_n_epochs(model,
                                                    loader_train, loader_val, L2_regularizer,L1_regularizer,
                                                    learning_rate, device, n_epoch, loss_weights,
                                                    save_weights = save_weights, output_dir = output_dir)

                                        resultsdf.loc[current_train-1] = [kernel,d_layer,dropoutRate,act_fn,L2_regularizer,L1_regularizer,
                                                learning_rate,
                                                lstm_units, n_lstm_layers, n_parameters,
                                                Train_loss, Train_accuracy, Val_loss, Val_accuracy]

                                        # Save a checkpoint                  
                                        df_filename = output_dir + 'df_checkpoint.pkl'
                                        resultsdf.to_pickle(df_filename)
                                        print(f'Checkpoint dataframe saved in: {df_filename}' )    

    elif model_type == 'CNN':
        # Hyperparameters to tune
        kernels = [3]
        d_layers = [128]
        dropoutRates = [0.3]
        L2_regularizers = [1e-6]
        L1_regularizers = [0]

        # Fixed
        act_fn=nn.ReLU
        learning_rates =  [5e-6]

        #save result
        resultsdf = pd.DataFrame(columns=['kernel','d_layer','dropoutRate','act_fn','L2_regularizer','L1_regularizer',
                                       'learning_rate', 'n_parameters',
                                       'Train_loss','Train_accuracy','Test_loss','Test_accuracy'])
        current_train = 0
        total_parameters = len(kernels)*len(d_layers)*len(dropoutRates)*len(L2_regularizers)*len(L1_regularizers)*len(learning_rates)
                           

        for kernel in kernels:
            for d_layer in d_layers:
                for dropoutRate in dropoutRates:
                    for L2_regularizer in L2_regularizers:
                        for L1_regularizer in L1_regularizers:
                            for learning_rate in learning_rates:
                                    current_train = current_train + 1
                                    print('\n--------------------------------------')
                                    print('\tSTART TRAIN n. '+
                                        f'{current_train}/{total_parameters}')
                                    print('---------------------------------------\n')

                                    model = Net_base_CNN(act_fn=act_fn,kernel=kernel,
                                            d_layer=d_layer,n_channels=n_channels,n_classes=n_classes,
                                            dropoutRate=dropoutRate)
                                    model = model.to(device)

                                    # compute the number of parameters in the model
                                    n_parameters = count_parameters(model)
                                    print('number of paramaters {}'.format(n_parameters))


                                    Train_loss, Train_accuracy, \
                                    Val_loss, Val_accuracy = train_for_n_epochs(model,
                                                loader_train, loader_val, L2_regularizer,L1_regularizer,
                                                learning_rate, device, n_epoch, loss_weights,
                                                save_weights = save_weights, output_dir = output_dir)
                                    
                                    resultsdf.loc[current_train-1] = [kernel,d_layer,dropoutRate,act_fn,L2_regularizer,L1_regularizer,
                                             learning_rate, n_parameters,
                                             Train_loss, Train_accuracy, Val_loss, Val_accuracy]  

    elif model_type == 'FC':
        # Hyperparameters to tune
        d_layers = [256]
        dropoutRates = [0.3]
        L2_regularizers = [1e-5]
        L1_regularizers = [0]

        # Fixed
        act_fn=nn.ReLU
        learning_rates =  [1e-05]

        #save result
        resultsdf = pd.DataFrame(columns=['d_layer','dropoutRate','act_fn','L2_regularizer','L1_regularizer',
                                       'learning_rate', 'n_parameters',
                                       'Train_loss','Train_accuracy','Test_loss','Test_accuracy'])
        current_train = 0
        total_parameters = len(d_layers)*len(dropoutRates)*len(L2_regularizers)*len(L1_regularizers)*len(learning_rates)
                           

        for d_layer in d_layers:
            for dropoutRate in dropoutRates:
                for L2_regularizer in L2_regularizers:
                    for L1_regularizer in L1_regularizers:
                        for learning_rate in learning_rates:
                                current_train = current_train + 1
                                print('\n--------------------------------------')
                                print('\tSTART TRAIN n. '+
                                    f'{current_train}/{total_parameters}')
                                print('---------------------------------------\n')

                                model = Net_base_FC(act_fn=act_fn,
                                        d_layer=d_layer,n_channels=n_channels,n_classes=n_classes,n_frames = n_frames,
                                        dropoutRate=dropoutRate)
                                model = model.to(device)

                                # compute the number of parameters in the model
                                n_parameters = count_parameters(model)
                                print('number of paramaters {}'.format(n_parameters))

                                Train_loss, Train_accuracy, \
                                Val_loss, Val_accuracy = train_for_n_epochs(model,
                                            loader_train, loader_val, L2_regularizer,L1_regularizer,
                                            learning_rate, device, n_epoch, loss_weights,
                                            save_weights = save_weights, output_dir = output_dir)
                            
                                resultsdf.loc[current_train-1] = [d_layer,dropoutRate,act_fn,L2_regularizer,L1_regularizer,
                                         learning_rate, n_parameters,
                                         Train_loss, Train_accuracy, Val_loss, Val_accuracy]  
   
    else:
        raise ValueError("Model not supported []")

    return resultsdf


