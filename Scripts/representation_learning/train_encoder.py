import sys
sys.path.append('../')
from dreamUtils import *
from data_augmentation import *
from split_reformat_data import *
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
from util_train_encoder import *


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

from sklearn.svm import SVC




if __name__ == '__main__':
    exp = 'LSTM_1frames'
    gridsearch = [1,2,3,4] #[1,2,3,4]


    subjects = ['H009','H018','H019','H021','H025','H026','H033','H035','H048','H050','H051','H054','H055','H057','H060','H061']

	# hyperparameters to tune
    Cs = np.logspace(-2, 10, 5)
    gammas = ['auto']#np.logspace(-9, 3, 3)
    kernels = ['linear','poly','rbf']
    degrees = [1]


	# train SVM
    resultsdf_subject = train_SVM(gridsearch, subjects, exp, Cs, gammas, kernels, degrees)

	# Save a dataframe for each subject with grid search results                   
    df_filename = 'Results/gridsearch/' + exp + '/train_encoder1'
    resultsdf_subject.to_pickle(df_filename)
    print(f'Dataframe saved in: {df_filename}' )




