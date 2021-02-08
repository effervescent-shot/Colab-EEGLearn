import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch as torch

def plot_loss_accuracy(Train_loss, Train_accuracy, Validation_loss, Validation_accuracy):
    '''plot train/validation loss and accuracy
    '''
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(1, 2,1)
    ax2 = fig.add_subplot(1, 2,2)

    ax1.plot(Train_loss,label='train loss')
    ax1.plot(Validation_loss,label='validation loss')
    ax1.legend()
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')

    ax2.plot(Train_accuracy,label='train accuracy')
    ax2.plot(Validation_accuracy,label='validation accuracy')
    ax2.legend()
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    plt.show()

def compute_accuracy(model, data_loader, device):
    """
    this function is used to compute the loss and accuracy on data without updating gradients.
    """
    model.eval()
    correct = 0
    total = 0

    Predicted = torch.zeros(0,dtype=torch.int64)
    Target = torch.zeros(0,dtype=torch.int64)
     
    with torch.no_grad():       
        for batch_idx, (_input, _target) in enumerate(data_loader): 
            #use GPU
            _input, _target = _input.to(device), _target.to(device)            
            output = model(_input)

            #compute the accuracy of Validation            
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
      
    return accuracy


def predict_single_aw_with_majority_voting(x, model, device):
    """
    Auxiliary function for predict_with_majority_voting(...)
    
    x are the videos coming from a SINGLE awakening.
    x is numpy array, shape: (n_videos, n_frames, 32, 32, n_channels)
    """
    #convert to numpy
    x = torch.from_numpy(x)
    
    #reshape them 
    x = x.permute(0,4,1,2,3)
    
    #change the type
    x = x.type(torch.float)
    x = x.to(device)
    
    #predict
    output = model(x)
    
    #to numpy
    out = output.data.cpu().numpy()
    
    ############ get a binary outcome ###########
    
    #first approach: sum the outcomes and then decide
    y_pred1 = out.sum(axis=0)
    y_pred1 = (y_pred1[1] > y_pred1[0])*1
    
    #second approach: decide and then take the majority
    y_pred2 = np.sum((out[:,1] > out[:,0]) * 1)/out.shape[0]
    y_pred2 = (y_pred2 > 0.5) * 1
    
    return y_pred1, y_pred2


def predict_with_majority_voting(data, labels, model, device):
    """
    Use majority voting to predict. If at least 50% of videos coming
    from one awakening have a certain prediction, the awakening get
    that prediction.
    
    Computed in 2 ways:
    method1 -> we compute the output (as probabilities) of the prediction applied to each 
        video, then we sum all the predictions (as probabilities) and take the maximum
    method2 -> we predict a label for each video and then we take the majority
        as the final prediction for the awakening
        
    Parameters
    ----------
    data : numpy array
        videos of several awakenings. shape = (n_awakenings, n_videos, n_frames, n_channels, 32, 32)
    labels : numpy array
        are the labels of the awakenings: shape (n_awakenings)
    model : pytorch model
        pre-trained model
    device : torch device
    
    Returns
    -------
    acc1 : accuracy computed with method 1 (see above)
    acc2 : accuracy computed with method 2 (see above)
    """
    #number of awakenings
    n_aw = labels.shape[0]
    
    #initialize accuracies
    acc1 = 0
    acc2 = 0
    
    #predict each awakening separately
    for i in range(n_aw):
        tmp = np.moveaxis(data[i], 2, -1)
        y1, y2 = predict_single_aw_with_majority_voting(tmp, model,device)
        acc1 = acc1 + (y1 == labels[i])*1
        acc2 = acc2 + (y2 == labels[i])*1
        
    acc1 = acc1 / n_aw
    acc2 = acc2 / n_aw
    
    return acc1*100, acc2*100
