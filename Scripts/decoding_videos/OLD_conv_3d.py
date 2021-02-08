### OLD APPROACH: 3D CONV

import os, sys, ast
sys.path.append('../eeg-dreams/Scripts')

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

import numpy as np
import pandas as pd

from split_reformat_data import *
from baseline_model import *

class Net_base_3Dconv(nn.Module):    
    def __init__(self, act_fn,kernel,d_layer,nb_channels,dropoutRate,padding):
        super(Net_base_3Dconv, self).__init__()
        #layer 1
        self.conv1 = nn.Sequential(nn.Conv3d(nb_channels,32,kernel_size=kernel),
                                   act_fn()) 
        self.conv2 = nn.Sequential(nn.Conv3d(32,32,kernel_size=kernel,padding = padding),
                                   act_fn()) 
        self.conv3 = nn.Sequential(nn.Conv3d(32,32,kernel_size=kernel,padding = padding),
                                   act_fn()) 
        self.conv4 = nn.Sequential(nn.Conv3d(32,32,kernel_size=kernel,padding = padding),
                                   act_fn())         
        self.maxpool1 = nn.MaxPool3d(kernel_size = (1,2,2))
        
        #layer 2
        self.conv5 = nn.Sequential(nn.Conv3d(32,64,kernel_size=kernel,padding = padding),
                                   act_fn()) 
        self.conv6 = nn.Sequential(nn.Conv3d(64,64,kernel_size=kernel,padding = padding),
                                   act_fn())         
        self.maxpool2 = nn.MaxPool3d(kernel_size = (2,2,2))
        
        # before the flatten
        self.cnn = nn.Sequential(self.conv1,self.conv2,self.conv3,self.conv4,
                                self.maxpool1, self.conv5,self.conv6,self.maxpool2)
        #
        self.fc1 = nn.Linear(self.last_layer_size(), d_layer)  
        self.fc2 = nn.Linear(d_layer, nb_channels) 
        self.dropout = nn.Dropout(p = dropoutRate)
        
    def last_layer_size(self):    
        # run dummy
        with torch.no_grad():
            x = torch.zeros([32, 2, 10, 32, 32])# dummy input with right shape
            y = self.cnn(x)
            return y.size(1)*y.size(2)*y.size(3)*y.size(4)
    
    def forward(self, x):
        #set_trace()       
        x = self.cnn(x)        
        #flatten
        #flatten_size = x.size(1)*x.size(2)*x.size(3)*x.size(4)
        x = x.view(x.shape[0], -1)
        
        #fc        
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x    


# Stolen from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def define_optim(model,L2_regularizer,learning_rate,optim_func=torch.optim.Adam):
    params = [
        {
            'params': [value],
            'name': key,
            'weight_decay': L2_regularizer if 'conv' in key or 'fc' in key else 0,
        }
        for key, value in model.named_parameters()
    ]
    optimizer = optim_func(
        params, lr=learning_rate
    ) 
    return optimizer

    torch.optim.Adam


def train_model(model,loader_train,L2_regularizer,learning_rate,device):
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = define_optim(model,L2_regularizer,learning_rate,optim_func=torch.optim.Adam)
    #optimizer = torch.optim.SGD(model.parameters(),momentum=0.9, lr = 0.01) 

    sum_loss = 0
    total = 0
    correct = 0
        
        
    for batch_idx, (_input, _target) in enumerate(loader_train):            
       
        #use GPU
        _input, _target = _input.to(device), _target.to(device)
        
        output = model(_input)
        loss = criterion(output, _target)
        model.zero_grad()
        loss.backward()
        optimizer.step()
            
        #conpute the loss
        sum_loss += loss.item() 
        loss_average = sum_loss/(batch_idx + 1)
            
        #compute the accuracy of train
        _, predicted = output.max(1)                        
        total += _target.size(0)
        correct += predicted.eq(_target).sum().item()
        accuracy = 100.*correct/total
    return loss_average, accuracy



def compute_nb_errors(model, loader_test,L2_regularizer,learning_rate,device):
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = define_optim(model,L2_regularizer,learning_rate,optim_func=torch.optim.Adam)

    sum_loss = 0
    total = 0
    correct = 0
     
    with torch.no_grad():       
        for batch_idx, (_input, _target) in enumerate(loader_test): 
            #use GPU
            _input, _target = _input.to(device), _target.to(device)
             
            output = model(_input)
            loss = criterion(output, _target)
            
            #conpute the loss
            sum_loss += loss.item() 
            loss_average = sum_loss/(batch_idx+1)

            #compute the accuracy of test            
            _, predicted = output.data.max(1)
            total += _target.size(0)
            correct += predicted.eq(_target).sum().item()
            accuracy = 100.*correct/total
        
    return loss_average, accuracy


def load_data(mini_batch_size):
    # Load data
    filename = '../dream_data_manipulated/10_10_FFT_SW_videos_normalized.npz'
    print('Loading data from:', filename)
    video = np.load(filename)
    data = video['videos']
    labels = video['labels']

    #
    labels = classes3_to_classes2(labels)
    X, Y = reformat_data_labels(data, labels)

    #
    X, Y = subsampling_labels(X, Y, shuffle = True, seed = 1)

    #
    ratio = 0.8
    x_train, x_test, y_train, y_test = split_data(X, Y, ratio=ratio, seed = 333)
    print('Train set is subdivided in train and test with ratio of:',ratio)

    #
    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    #
    x_train = x_train.permute(0,4,1,2,3)
    x_test = x_test.permute(0,4,1,2,3)

    #
    x_train = x_train.type(torch.float)
    x_test = x_test.type(torch.float)

    #
    y_train = y_train.type(torch.LongTensor)
    y_test = y_test.type(torch.LongTensor)

    # Create dataset from several tensors with matching first dimension
    # Samples will be drawn from the first dimension (rows)
    dataset_train = TensorDataset(x_train, y_train)
    dataset_test = TensorDataset(x_test, y_test)

    mini_batch_size = 32
    # Create a data loader from the dataset
    # Type of sampling and batch size are specified at this step
    loader_train = DataLoader(dataset_train, batch_size= mini_batch_size,shuffle=1)
    loader_test = DataLoader(dataset_test, batch_size= mini_batch_size, shuffle=1)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(n_epoch,loader_train,loader_test):
    #using GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #define hyperparameters    
    kernel = (2, 5, 5)#or kernel = (2, 3, 3)
    padding = (2,3,3)
    d_layer = 128
    dropoutRate = 0.3
    nb_channels = 2
    act_fn=nn.ReLU #nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid


    L2_regularizer = 0.001
    learning_rate = 0.00005


    model = Net_base_3Dconv(act_fn=act_fn, kernel=kernel,
                            d_layer=d_layer,
                            nb_channels=nb_channels,dropoutRate=dropoutRate, padding=padding)

    model = model.to(device)

    early_stopping = EarlyStopping(patience=7, verbose=True)


    for k in range(n_epoch):
        train_loss, train_accuracy = train_model(model, loader_train,L2_regularizer,learning_rate,device)
        test_loss, test_accuracy = compute_nb_errors(model, loader_test,L2_regularizer,learning_rate,device)

        print('Epoch {}/{} train_loss {} train_accuracy {}% test_loss {} test_accuracy {}%'
            .format(k,n_epoch,round(train_loss,2), round(train_accuracy,2), round(test_loss,2), round(test_accuracy,2)))

        early_stopping(test_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break      
    




def train_tune(n_epoch,loader_train,loader_test):
    #using GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    #define hyperparameters to tune with
    kernels = [(2, 5, 5),(2, 3, 3)]
    paddings = [(2,3,3),(2,2,2)]
    d_layers = [128,512]
    dropoutRates = [0,0.3,0.5,0.8]
    act_fns=[nn.ReLU, nn.LeakyReLU]
    L2_regularizers = [1e-3,1e-4,1e-5]
    learning_rates = [0.00005,0.0005,0.005]

    patience = 15


    #fixed paramaters
    nb_channels = 2
    
    #save result
    resultsdf = pd.DataFrame(columns=['best_test_loss', 'best_test_accuracy','kernel','padding','d_layer','dropoutRate','act_fn','L2_regularizer',
                                       'learning_rate','Train_loss','Train_accuracy','Test_loss','Test_accuracy'])

    
    current_train = 0

    for kernel in kernels:
        for padding in paddings:
            for d_layer in d_layers:
                for dropoutRate in dropoutRates:
                    for act_fn in act_fns:
                        for L2_regularizer in L2_regularizers:
                            for learning_rate in learning_rates:
                                current_train += 1

                                print('current training {}, kernel {}, padding {}, d_layer {}, dropoutRate {}, act_fn {}, L2_regularizer {}, learning_rate {}'.format(
                                    current_train, kernel, padding, d_layer, dropoutRate, act_fn, L2_regularizer, learning_rate
                                )
                                       )

                                # early_stopping = EarlyStopping(patience, verbose=True)

                                model = Net_base_3Dconv(act_fn=act_fn, kernel=kernel,
                                    d_layer=d_layer, nb_channels=nb_channels,dropoutRate=dropoutRate, padding=padding)

                                model = model.to(device)

                                Train_loss = []
                                Train_accuracy = []
                                Test_loss = []
                                Test_accuracy = []
                                best_validation_accu = 0
                                for k in range(n_epoch):
                                    train_loss, train_accuracy = train_model(model, loader_train,L2_regularizer,learning_rate,device)
                                    test_loss, test_accuracy = compute_nb_errors(model, loader_test,L2_regularizer,learning_rate,device)
                                    
                              
                                    print('Epoch {}/{} train_loss {} train_accuracy {}% test_loss {} test_accuracy {}%'
                                        .format(k,n_epoch,round(train_loss,2), round(train_accuracy,2), round(test_loss,2), round(test_accuracy,2)))
                                    
                                    Train_loss.append(train_loss)
                                    Train_accuracy.append(train_accuracy)
                                    Test_loss.append(test_loss)
                                    Test_accuracy.append(test_accuracy)                                
 
                                    # early_stopping(test_loss, model)
                                    # if early_stopping.early_stop:
                                    #     print("Early stopping")
                                    #     break 

                                    if test_accuracy > best_validation_accu:   # early_stoping
                                        stop_count = 0
                                        eraly_stoping_epoch = n_epoch
                                        best_validation_accu = test_accuracy
                                    else:
                                        stop_count += 1
                                        if stop_count >= 15: # stop training if val_acc dose not imporve for over 15 epochs
                                            break


                                resultsdf.loc[current_train-1] = [test_loss,test_accuracy,kernel,padding,d_layer,dropoutRate,
                                act_fn,L2_regularizer,learning_rate,Train_loss,Train_accuracy,Test_loss,Test_accuracy]

    resultsdf.to_pickle('dataframe_summary_gridsearch.pkl')



                                


if __name__ == "__main__":
    train(300)
    
    

