import torch as torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

import numpy as np
import pandas as pd


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


def train_model_RL(model,loader_train,L2_regularizer,learning_rate,L1_regularizer,device):
    """
    This funcion is use to rain for ONE epoch.

    parameters:
    model: the model.
    loader_train: dataloader for train.
    L2_regularizer: the restrain for L2_regularizer.
    learning_rate: learning rate for optimizer.
    device: GPU
    L1_regularizer: the restrain for L2 regularizer
    """

    model.train()
    
    criterion = nn.MSELoss()
    optimizer = define_optim(model,L2_regularizer,learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(),momentum=0.9, lr = 0.01) 

    sum_loss = 0
    total = 0
    correct = 0
        
        
    for batch_idx, (_input, _target) in enumerate(loader_train):  
       #use GPU
        _input, _target = _input.to(device), _target.to(device)
        _, output  = model(_input)


        loss = criterion(output, _target)
        # update the gradients
        model.zero_grad()
        loss.backward()
        optimizer.step()
            
        # sum and average the loss 
        sum_loss += loss.item() 
        loss_average = sum_loss/(batch_idx + 1)
    return loss_average



def compute_nb_errors_RL(model, loader_test,L2_regularizer,learning_rate,L1_regularizer,device):
    """
    this function is used to compute the loss and accuracy on test data without updating gradients.
    """

    model.eval()
    
    criterion = nn.MSELoss()
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
            _, output  = model(_input)

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
        
    return loss_average



def get_encoder(x_dataloader,model,GPU):
    """
    this function is used to convet the orginal data to the encoder vectors 
    by passing data to the pre-trained model.

    :param x_dataloader: the dataloader.
    :parm model: the model
    :parm GPU: GPU

    :output:
    x_encoder
    """
    # Using GPU
    if GPU != -1:
        gpu_string = f'cuda:{GPU}'
        device = torch.device(gpu_string if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    x_encoder = []
    for i, x_data in enumerate(x_dataloader):
        x_data = x_data.to(device)
        x_encoder_batch,_ =model(x_data)
        tmp = x_encoder_batch.data.cpu().numpy()

        if i == 0:
            x_encoder = tmp
        else:
            x_encoder = np.vstack((x_encoder,tmp))
            
    return x_encoder
 



