import sys
sys.path.append('../')
from dreamUtils import * #tf_dreamUtils
from data_augmentation import *
from split_reformat_data import * #Luo_split_reformat_data
from helpers import *

import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

import numpy as np
import pandas as pd

# pytorch stuff
from util_RL import *
from pytorch_models_RL import *
from training_with_pytorch_RL import *
from utils_pytorch_RL import *


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score


def grid_search_logistic(x_train, y_train, x_val, y_val ,x_te, y_te,Cs =[1], degrees = [1]):
    '''Grid search using logistic regression of sklearn.
    
        Parameters
    ----------
    x_train, x_val : 2D numpy array
        Training and validation sets
    y_train, y_val : 1D numpy array
        Training and validation labels
    Cs : list of floats, optional
        Regularization constants to try (inverse, as in 
        sklearn.linear_model.LogisticRegression)
    degrees : list of integers, optional
        Polynomial augmentation degrees (with mixed terms) to try.
        
        Returns
    -------    
    resultsdf : pandas dataframe
        Dataframe with the results of the grid search
    '''
    resultsdf = pd.DataFrame(columns=['C','degree','train_acc','val_acc','te_acc','majority_voting_prob_acc','majority_voting_navie_acc'])
    
    current_train = 1
    total_trs = len(Cs)*len(degrees)
    for degree in degrees:
        for C in Cs:
            print(f'Start training: {current_train}/{total_trs} ' +\
                                                f'(C = {C}, degree = {degree})')
            
            # Augment data
            poly = PolynomialFeatures(degree)
            tr_augmented = poly.fit_transform(x_train)
            val_augmented = poly.fit_transform(x_val)
            te_augmented = poly.fit_transform(x_te)
            
            # Build and train the model
            logreg = linear_model.LogisticRegression(C=C,solver='liblinear')
            logreg = logreg.fit(tr_augmented, y_train)
            
            # Compute accuracies
            z = logreg.predict(tr_augmented)
            tr_acc = np.sum(y_train == z) / z.shape[0]
            z = logreg.predict(val_augmented)
            val_acc = np.sum(y_val == z) / z.shape[0]
            z = logreg.predict(te_augmented)
            te_acc = np.sum(y_te == z) / z.shape[0]

            # majority voting
            timepoints = 237
            video_size = 10
            slide = 3      
            majority_voting_prob_acc = majority_voting_te(te_augmented,y_te,timepoints,video_size,slide ,logreg,method='probability')
            majority_voting_navie_acc = majority_voting_te(te_augmented,y_te,timepoints,video_size,slide ,logreg,method='navie')
           
            # Update df
            resultsdf.loc[current_train-1] = [C, degree, tr_acc, val_acc,te_acc,majority_voting_prob_acc,majority_voting_navie_acc]
            current_train = current_train +  1
            
    return resultsdf


def grid_search_SVM(x_train, y_train, x_val, y_val ,x_te, y_te,Cs,gammas ,kernels,degrees):
    '''Grid search using logistic regression of sklearn.
    
        Parameters
    ----------
    x_train, x_val : 2D numpy array
        Training and validation sets
    y_train, y_val : 1D numpy array
        Training and validation labels
    Cs : list of floats, optional
        Regularization constants to try (inverse, as in 
        sklearn.linear_model.LogisticRegression)
    degrees : list of integers, optional
        Polynomial augmentation degrees (with mixed terms) to try.
        
        Returns
    -------    
    resultsdf : pandas dataframe
        Dataframe with the results of the grid search
    '''
    
    resultsdf = pd.DataFrame(columns=['C','gamma','kernel','degree','train_acc','val_acc','te_acc','majority_voting_prob_acc','CM_te','tr_balanced_acc','val_balanced_acc','te_balanced_acc','majority_voting_prob_balanced_acc','y_tr','y_val','y_te','predicted_tr','predicted_val','predicted_te','Y_pred_MJ','Y_awanking_MJ'])
    
    current_train = 1
    total_trs = len(Cs)*len(degrees)*len(kernels)*len(gammas)
    for degree in degrees:
        for C in Cs:
            for gamma in gammas:
                for kernel in kernels:

                    print(f'Start training: {current_train}/{total_trs} ' +\
                                                    f'(C = {C}, gamma = {gamma}, kernel = {kernel}, degree = {degree})')
                    
                    # Augment data
                    poly = PolynomialFeatures(degree)
                    tr_augmented = poly.fit_transform(x_train)
                    val_augmented = poly.fit_transform(x_val)
                    te_augmented = poly.fit_transform(x_te)


                    # Build and train the model            
                    clf = SVC(C = C, gamma=gamma, kernel = kernel, degree = degree, probability=True)
                    svm = clf.fit(tr_augmented, y_train)

                    # Compute accuracies
                    z = svm.predict(tr_augmented)
                    predicted_tr = z
                    tr_acc = np.sum(y_train == z) / z.shape[0]
                    tr_balanced_acc = balanced_accuracy_score(y_train,z)

                    z = svm.predict(val_augmented)
                    predicted_val = z
                    val_acc = np.sum(y_val == z) / z.shape[0]
                    val_balanced_acc = balanced_accuracy_score(y_val,z)

                    z = svm.predict(te_augmented)
                    predicted_te = z
                    te_acc = np.sum(y_te == z) / z.shape[0]
                    te_balanced_acc = balanced_accuracy_score(y_te,z)
                    CM_te = confusion_matrix(y_te, z) # confusion matrix

                  
                    # majority voting
                    timepoints = 237
                    video_size = 10
                    slide = 3      
                    majority_voting_prob_acc, majority_voting_prob_balanced_acc,Y_pred_MJ,Y_awanking_MJ= majority_voting_te(te_augmented,y_te,timepoints,video_size,slide ,svm,method='probability')
                   # majority_voting_navie_acc, majority_voting_navie_balanced_acc,Y_pred_MJ_navie,Y_awanking_MJ_prob_naive = majority_voting_te(te_augmented,y_te,timepoints,video_size,slide ,svm,method='navie')                    
                    

                    # Update df
                    resultsdf.loc[current_train-1] = [C, gamma, kernel, degree, tr_acc, val_acc,te_acc, majority_voting_prob_acc,CM_te,tr_balanced_acc,val_balanced_acc,te_balanced_acc,majority_voting_prob_balanced_acc,y_train,y_val,y_te,predicted_tr,predicted_val,predicted_te,Y_pred_MJ,Y_awanking_MJ]
                    current_train = current_train +  1
                    
    return resultsdf,svm


def grid_search_randomforest(x_train, y_train, x_val, y_val ,x_te, y_te,degrees,n_estimators,max_depths,criterions,weighted_0s,random_state=0):
    '''Grid search using logistic regression of sklearn.
    
        Parameters
    ----------
    x_train, x_val : 2D numpy array
        Training and validation sets
    y_train, y_val : 1D numpy array
        Training and validation labels
    Cs : list of floats, optional
        Regularization constants to try (inverse, as in 
        sklearn.linear_model.LogisticRegression)
    degrees : list of integers, optional
        Polynomial augmentation degrees (with mixed terms) to try.
        
        Returns
    -------    
    resultsdf : pandas dataframe
        Dataframe with the results of the grid search
    '''
    resultsdf = pd.DataFrame(columns=['degrees','n_estimators','max_depths','criterions','weighted_0s','train_acc','val_acc','te_acc','majority_voting_prob_acc','CM_te','tr_balanced_acc','val_balanced_acc','te_balanced_acc','majority_voting_prob_balanced_acc','y_tr','y_val','y_te','predicted_tr','predicted_val','predicted_te','Y_pred_MJ','Y_awanking_MJ'])
    
    current_train = 1
    total_trs = len(degrees)*len(n_estimators)*len(max_depths)*len(criterions)*len(weighted_0s)
    for degree in degrees:
        for n_estimator in n_estimators:
            for max_depth in max_depths:
                for criterion in criterions:
                    for weighted_0 in weighted_0s:
            
                        print(f'Start training: {current_train}/{total_trs} ' +\
                                                            f'(degree = {degree}, n_estimator = {n_estimator}),max_depth = {max_depth},criterion = {criterion},weighted_0 = {weighted_0}')

                        # Augment data
                        poly = PolynomialFeatures(degree)
                        tr_augmented = poly.fit_transform(x_train)
                        val_augmented = poly.fit_transform(x_val)
                        te_augmented = poly.fit_transform(x_te)

                        # Build and train the model
                        RF = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, criterion=criterion,random_state=0,class_weight={0: weighted_0})
                        RF = RF.fit(tr_augmented, y_train)

                        # Compute accuracies
                        z = RF.predict(tr_augmented)
                        predicted_tr = z
                        tr_acc = np.sum(y_train == z) / z.shape[0]
                        tr_balanced_acc = balanced_accuracy_score(y_train,z)

                        z = RF.predict(val_augmented)
                        predicted_val = z
                        val_acc = np.sum(y_val == z) / z.shape[0]
                        val_balanced_acc = balanced_accuracy_score(y_val,z)

                        z = RF.predict(te_augmented)
                        predicted_te = z
                        te_acc = np.sum(y_te == z) / z.shape[0]
                        te_balanced_acc = balanced_accuracy_score(y_te,z)
                        CM_te = confusion_matrix(y_te, z) # confusion matrix


                        # majority voting
                        timepoints = 237
                        video_size = 10
                        slide = 3      
                        majority_voting_prob_acc, majority_voting_prob_balanced_acc,Y_pred_MJ,Y_awanking_MJ= majority_voting_te(te_augmented,y_te,timepoints,video_size,slide ,RF,method='probability')
                       # majority_voting_navie_acc, majority_voting_navie_balanced_acc,Y_pred_MJ_navie,Y_awanking_MJ_prob_naive = majority_voting_te(te_augmented,y_te,timepoints,video_size,slide ,RF,method='navie')                    


                        # Update df
                        resultsdf.loc[current_train-1] = [degree, n_estimator,max_depth, criterion, weighted_0, tr_acc, val_acc,te_acc, majority_voting_prob_acc,CM_te,tr_balanced_acc,val_balanced_acc,te_balanced_acc,majority_voting_prob_balanced_acc,y_train,y_val,y_te,predicted_tr,predicted_val,predicted_te,Y_pred_MJ,Y_awanking_MJ]
                        current_train = current_train +  1


    return resultsdf, RF

def gen_train_val(data):

    # load data
    x_train_encoder = data['x_train_encoder']
    x_val_encoder = data['x_val_encoder']
    x_test_encoder = data['x_test_encoder']
    label_classes_train = data['label_classes_train']
    label_classes_val = data['label_classes_val']
    label_classes_test = data['label_classes_test']
    sleep_stage_train = data['sleep_stage_train']
    sleep_stage_val = data['sleep_stage_val']
    sleep_stage_test = data['sleep_stage_test']
    
    # convert labels
    # # conver class labels: DE, DEWR --> DE
    label_classes_train[label_classes_train == 2] =1
    label_classes_val[label_classes_val == 2] =1
    label_classes_test[label_classes_test == 2] =1

    # convert sleep stage
    # # RAM 1, NRAM = 0
    sleep_stage_train[sleep_stage_train != 4] = 0
    sleep_stage_train[sleep_stage_train == 4] = 1
    sleep_stage_val[sleep_stage_val != 4] = 0
    sleep_stage_val[sleep_stage_val == 4] = 1
    sleep_stage_test[sleep_stage_test != 4] = 0
    sleep_stage_test[sleep_stage_test == 4] = 1   
    

    # one-hot coding on sleep stage    
    sleep_stage = np.hstack((sleep_stage_train,sleep_stage_val,sleep_stage_test))
    sleep_stage_dummy = pd.get_dummies(sleep_stage).values
    
    a1 = sleep_stage_train.shape[0]
    a2 = sleep_stage_train.shape[0] + sleep_stage_val.shape[0]
    a3 = sleep_stage_train.shape[0] + sleep_stage_val.shape[0] + sleep_stage_test.shape[0]
    sleep_stage_train_dummy = sleep_stage_dummy[0:a1]
    sleep_stage_val_dummy = sleep_stage_dummy[a1:a2]
    sleep_stage_test_dummy = sleep_stage_dummy[a2:a3]
    # add one-hot coding sleep stage
    x_train_encoder_sleep_stage = np.hstack((x_train_encoder,sleep_stage_train_dummy))
    x_val_encoder_sleep_stage = np.hstack((x_val_encoder,sleep_stage_val_dummy))
    x_test_encoder_sleep_stage = np.hstack((x_test_encoder,sleep_stage_test_dummy))
    
    #set train/test
    x_tr = x_train_encoder_sleep_stage
    y_tr = label_classes_train
    x_val = x_val_encoder_sleep_stage
    y_val = label_classes_val
    x_te = x_test_encoder_sleep_stage
    y_te = label_classes_test
    
    # balanced the train set
    x_tr, y_tr = subsampling_labels(x_tr, y_tr, shuffle = True, seed = 1)    
    
    return x_tr, y_tr, x_val, y_val, x_te, y_te


def train_SVM(gridsearch, subjects, exp, Cs, gammas, kernels, degrees):

    resultsdf_subject = pd.DataFrame(columns=['C','gamma','kernel','degree', 'train_acc','val_acc','te_acc','subject','grid_search'])
    for i in gridsearch:
        for subject in subjects:
            print(subject)
            print(i)        
            
            data = np.load('Results/gridsearch/' + exp + '/weights' + subject + '/encoder_data_gridsearch' + str(i) + '.npz')
            x_tr, y_tr, x_val, y_val, x_te, y_te = gen_train_val(data)     
            
            resultsdf = grid_search_SVM(x_tr, y_tr, x_val, y_val, x_te, y_te,Cs = Cs,gammas= gammas, kernels= kernels,degrees = degrees)

            resultsdf['subject'] = subject
            resultsdf['grid_search'] = i
            resultsdf_subject = resultsdf_subject.append(resultsdf)

            print(resultsdf)
            print('#########################')

    return resultsdf_subject

def majority_voting_te(x_te,y_te,timepoints,video_size,slide ,model,method):
    
    N_after_videos = 1
    n_video_in_awaking = (timepoints-(video_size+N_after_videos))//slide + 1
    n_awaking = x_te.shape[0]/n_video_in_awaking
    

    acc1 = 0
    Y_pred = []
    Y_awanking = []
    for awaking in range(int(n_awaking)):
        awanking_x = x_te[awaking*n_video_in_awaking : (awaking+1)*n_video_in_awaking]
        awanking_y = np.unique(y_te[awaking*n_video_in_awaking : (awaking+1)*n_video_in_awaking])


        if method == 'probability':
            # Use the model on the data coming from aw
            probabilities = model.predict_proba(awanking_x)
            # Sum the probabilities and then decise
            probabilities = probabilities.sum(axis=0)
            y_pred1 = (probabilities[1] > probabilities[0])*1
            
        if method =='navie':
            predicted = model.predict(awanking_x)
            
            pred0 = len(np.where(predicted == 0)[0])
            pred1 = len(np.where(predicted == 1)[0])
            y_pred1 = (pred1 > pred0)*1
 
        # Update accuracy
        acc1 = acc1 + (y_pred1 == awanking_y)*1 
        Y_pred.append(y_pred1)  
        Y_awanking.append(awanking_y)
        

    balanced_acc = balanced_accuracy_score(Y_pred,Y_awanking)

        
    return acc1/n_awaking, balanced_acc,Y_pred,Y_awanking


