
## WARNING: NOT UPDATED WITH WEIGHT LOSSES AND DIFFERENT METRICS

import torch as torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from sklearn.metrics import balanced_accuracy_score

from torchcontrib.optim import SWA

import numpy as np
import pandas as pd


def dataloader_pytorch(x, y, mini_batch_size,shuffle=True):
    """
    This function take data x and label y to convert them as a dataloader.
    
    """   
    #convert to numpy
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    #reshape them 
    x = x.permute(0,4,1,2,3)
    
    #change the type
    x = x.type(torch.float)
    y = y.type(torch.LongTensor)
    
    #Dataloader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size= mini_batch_size,shuffle=True)   
    return dataloader


def dataloader_pytorch_RL(x, y, mini_batch_size,shuffle=True):
    """
    This function take data x and label y to convert them as a dataloader.
    
    """
    
    #convert to numpy
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    #reshape them 
    x = x.permute(0,2,1,3,4)
    
    #change the type
    x = x.type(torch.float)
    y = y.type(torch.float)
    
    #Dataloader
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size= mini_batch_size,shuffle=True)   
    return dataloader



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

    """
    This function is used to define the optimizer and together with L2 regularization.
    parameters: 
    model: the input model
    L2_regularizer: the lambda for L2 regularization.
    optim_func : the loss function. eg: torch.optim.Adam, torch.optim.Adadelta.
    learning_rate: the learning rate of the optimizer

    output:
    optimizer: the optimizer function that was used for training.
    """
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


def train_model(model,loader_train,L2_regularizer,learning_rate,L1_regularizer,
                device, loss_weights = [1,1]):
    """
    This funcion is use to rain for ONE epoch.

    parameters:
    model: the model.
    loader_train: dataloader for train.
    L2_regularizer: the restrain for L2_regularizer.
    learning_rate: learning rate for optimizer.
    device: GPU
    L1_regularizer: the restrain for L2 regularizer
    loss_weigths: weights that are assigned to each class during training
    """

    model.train()
            
    tensor_weights = torch.FloatTensor(loss_weights).to(device)
    criterion = nn.CrossEntropyLoss(tensor_weights)
    
    base_opt = define_optim(model,L2_regularizer,learning_rate)
    
    #SWA
    n_steps = len(loader_train.dataset) // loader_train.batch_size
    batch_size = loader_train.batch_size
#     optimizer = SWA(base_opt, swa_start = int(n_steps*0.75), swa_freq=1, swa_lr = 10*learning_rate)
    optimizer = SWA(base_opt, swa_start = 1, swa_freq=1, swa_lr = learning_rate)
    
    sum_loss = 0
    total = 0
    correct = 0
        
        
    for batch_idx, (_input, _target) in enumerate(loader_train):            
        #use GPU
        _input, _target = _input.to(device), _target.to(device)
        output = model(_input)

        loss = criterion(output, _target)
        
        # update the gradients
        model.zero_grad()
        loss.backward()
        optimizer.step()
 
        # sum and average the loss 
        sum_loss += loss.item() 
        loss_average = sum_loss/(batch_idx + 1)
   
        #compute the accuracy of train
        _, predicted = output.max(1)                        
        total += _target.size(0)
        correct += predicted.eq(_target).sum().item()
        accuracy = 100.*correct/total

        # concat all the predicted targets and true targers
        Predicted = torch.cat((Predicted,predicted.cpu()))
        Target = torch.cat((Target,_target.cpu()))

    # integrate the predicted and target    
    Predicted = Predicted.numpy()
    Target = Target.numpy()

    # Find dreaming and no dreaming indices
    ones_idx = np.where(Target == 1)[0]
    zeros_idx = np.where(Target == 0)[0]

    ones_acc = np.sum(Predicted[ones_idx]==1)/ones_idx.shape[0]
    zeros_acc = np.sum(Predicted[zeros_idx]==0)/zeros_idx.shape[0]
    balanced_acc = balanced_accuracy_score(Target,Predicted)

    optimizer.swap_swa_sgd()      
     
    return loss_average, [accuracy,ones_acc,zeros_acc,balanced_acc]




def compute_nb_errors(model, loader_test, L2_regularizer, learning_rate, 
                      L1_regularizer, device, loss_weights = [1,1]):
    """
    this function is used to compute the loss and accuracy on test data without 
    updating gradients.
    """

    model.eval()
    
    tensor_weights = torch.FloatTensor(loss_weights).to(device)
    criterion = nn.CrossEntropyLoss(tensor_weights)

    optimizer = define_optim(model,L2_regularizer,learning_rate,optim_func=torch.optim.Adam)

    sum_loss = 0
    total = 0
    correct = 0

    Predicted = torch.zeros(0,dtype=torch.int64)
    Target = torch.zeros(0,dtype=torch.int64)
     
    with torch.no_grad():       
        for batch_idx, (_input, _target) in enumerate(loader_test): 
            #use GPU
            _input, _target = _input.to(device), _target.to(device)            
            output = model(_input)

#             # compute the L1 weights
#             L1_regularization = 0
#             for param in model.parameters():
#                 L1_regularization += torch.norm(param, 1)
#             #conpute the loss   
#             loss = criterion(output, _target) + L1_regularization * L1_regularizer

            loss = criterion(output, _target)

            #sum and average the loss
            sum_loss += loss.item() 
            loss_average = sum_loss/(batch_idx+1)

            #compute the accuracy of test            
            _, predicted = output.data.max(1)
            total += _target.size(0)
            correct += predicted.eq(_target).sum().item()
            accuracy = 100.*correct/total

            # concat all the predicted targets and true targers
            Predicted = torch.cat((Predicted,predicted.cpu()))
            Target = torch.cat((Target,_target.cpu()))

        # integrate the predicted and target    
        Predicted = Predicted.numpy()
        Target = Target.numpy()
        
        # Find dreaming and no dreaming indices
        ones_idx = np.where(Target == 1)[0]
        zeros_idx = np.where(Target == 0)[0]

        ones_acc = np.sum(Predicted[ones_idx]==1)/ones_idx.shape[0]
        zeros_acc = np.sum(Predicted[zeros_idx]==0)/zeros_idx.shape[0]
        balanced_acc = balanced_accuracy_score(Target,Predicted)
        
    return loss_average, [accuracy,ones_acc,zeros_acc,balanced_acc]

def count_parameters(model):
    """
    this is the function that we could
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
