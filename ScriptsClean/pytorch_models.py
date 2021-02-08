import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd

def CNN_model(kernel, act_fn, n_channels, dropoutRate=0):
        ######################### CNN #########################
        ######################### layer 1 of cnn #########################
        #caculate the padding 'same'
        assert((kernel -1)%2  == 0)
        padding = int((kernel -1)/2)       
        conv1 = nn.Sequential(nn.Conv2d(n_channels, 32, kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn()) 
        frame_size = 32
        H_neurons = 32 #[batch_size, H_neurons, frame_size, frame_size]  
    
        conv2 = nn.Sequential(nn.Conv2d(32,32,kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn()) 
        frame_size = frame_size 
        H_neurons = 32 #[batch_size, H_neurons, frame_size, frame_size] 
                
        conv3 = nn.Sequential(nn.Conv2d(32,32,kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn()) 
        frame_size = frame_size 
        H_neurons = 32 #[batch_size, H_neurons, frame_size, frame_size] 
        
        conv4 = nn.Sequential(nn.Conv2d(32,32,kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn()) 
        frame_size = frame_size 
        H_neurons = 32 #[batch_size, H_neurons, frame_size, frame_size] 
                
        maxpool1 = nn.MaxPool2d(kernel_size = (2,2))
        frame_size = frame_size/2  #15
        H_neurons = 32 #[batch_size, H_neurons, frame_size, frame_size] 
                
        ######################### layer 2 of cnn #########################
        conv5 = nn.Sequential(nn.Conv2d(32,64,kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn()) 
        frame_size = frame_size  #15
        H_neurons = 64 #[batch_size, H_neurons, frame_size, frame_size] 
        
        conv6 = nn.Sequential(nn.Conv2d(64,64,kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn())  
        frame_size = frame_size  #15
        H_neurons = 64 #[batch_size, 64, 16, 16] 
        
        maxpool2 = nn.MaxPool2d(kernel_size = (2,2))
        frame_size = frame_size/2 #8
        H_neurons = 64 #[batch_size, 64, 8, 8] 
        
        ######################### layer 3 of cnn #########################
        conv7 = nn.Sequential(nn.Conv2d(64,128,kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn())   
        frame_size = frame_size #8
        H_neurons = 128 #[batch_size, 128, 8, 8] 
        
        maxpool3 = nn.MaxPool2d(kernel_size = (2,2)) 
        assert(frame_size % 2 == 0)
        frame_size = int(frame_size/2)  #4
        H_neurons = 128 #[batch_size, 128, 4, 4] 
        
                
        ### ---> integrate cnn layers <--- ####
        cnn = nn.Sequential(conv1,conv2,conv3,conv4,
                                maxpool1, conv5,conv6, maxpool2,
                                conv7, maxpool3 )
        
#        print('--------------------- ONE MORE DROPOUT!! -------------------')
#        cnn = nn.Sequential(conv1,conv2,conv3,conv4,
#                                 maxpool1, conv5,conv6, maxpool2,
#                                 conv7, maxpool3, dropout )
        
        return cnn, frame_size, H_neurons


class Net_base_2DConv(nn.Module):    
    def __init__(self, act_fn,kernel,d_layer,n_channels,n_classes,dropoutRate,n_frames,time_pool_size):
        """
        video size: [n_samples,n_channels,n_frames,32,32]
        image size: [batch_size, n_channles, 32, 32]
        
    
        :param n_channels: number of color channels
        :param n_frames: number of frames in the video
        :param kernel: the size of kernel, eg: 3 or 5 --> (3,3) or (5,5)
        :param act_fn: activation function, eg: nn.ReLU, nn.Sigmoid, nn.LeakyReLU
        :param time_pool_size: kernel size of maxpooling on the time domain. It should be exact divisor of time_pool_size.
                              eg: the n_frames is 10, we suggest to put n_frames as 2, 5 or 10.
        :param d_layer : the number of neurons in the last dense layer.
        :param dropoutRate : dropout rate.             
        """
        
        super(Net_base_2DConv, self).__init__()
        ######################### CNN #########################
        self.cnn, frame_size, H_neurons = CNN_model(kernel,act_fn,n_channels)       
        #after the stack of frames, the size of data becames: [n_samples, 128, 4, 4,n_frames]
        
        ######################### max pooling over frames #########################
        self.maxpool4 = nn.MaxPool3d(kernel_size = (1,1,time_pool_size)) #[n_samples, 128, 4, 4,n_frames]
        assert(n_frames % time_pool_size == 0)
        time_size = int(n_frames/time_pool_size)
        frame_size = frame_size
        H_neurons = 128 #[n_samples, H_neurons, frame_size, frame_size,time_size]
               
        ######################### fully connected layer #########################
        self.dropout = nn.Dropout(p = dropoutRate)
        self.fc1 = nn.Sequential(nn.Linear(H_neurons*frame_size*frame_size*time_size, d_layer),act_fn())
        self.fc2 = nn.Linear(d_layer, n_classes)
        self.FC = nn.Sequential(self.dropout,self.fc1,self.dropout,self.fc2)

        
    def forward(self, x): 
        #x shape: [n_samples,n_channels,n_frames,32,32]   
        #set_trace()
        convnets = []
        convnet = 0
        n_frames = x.shape[2]
        for i in range(n_frames):
            convnet = x[:,:,i,:,:]    #[n_samples,n_channels,32,32]        
            #cnn
            convnet = self.cnn(convnet) #[n_samples, 128, 4, 4]
            convnets.append(convnet)            
        convnets = torch.stack(convnets) #[n_frames, n_samples, 128, 4, 4]
        convnets = convnets.permute(1,2,3,4,0) #[n_samples, 128, 4, 4,n_frames]
        
        # max pooling over time(frames)
        convnets = self.maxpool4(convnets) #[n_samples, 128, 4, 4, n_frames/time_pool_size]
        
        # flatten and FC
        convnets = convnets.view(convnets.shape[0],-1)
        convnets = self.FC(convnets)           
        return convnets   

class Net_base_convpool_conv1d(nn.Module):        
    """
        video size: [n_samples,n_channels,n_frames,32,32]
        image size: [batch_size, n_channles, 32, 32]
        
        
        :param n_channels: number of color channels
        :param n_frames: number of frames in the video
        :param kernel: the size of kernel, eg: 3 or 5 --> (3,3) or (5,5)
        :param act_fn: activation function, eg: nn.ReLU, nn.Sigmoid, nn.LeakyReLU
        :param k_size_1d: kernel size of 1d conv(in time domain)
        :param d_layer : the number of neurons in the last dense layer.
        :param dropoutRate : dropout rate.
                
    """
    
    def __init__(self, act_fn, kernel, d_layer, n_channels, n_classes, dropoutRate, n_frames, k_size_1d):
        super(Net_base_convpool_conv1d, self).__init__()
        ######################### CNN #########################
        self.cnn, frame_size, H_neurons = CNN_model(kernel,act_fn,n_channels)          
        #after the stack of frames, the size of data becames: [n_samples, 128, 4, 4,n_frames]
        
        ######################### conv1d #########################
        self.conv8 = nn.Sequential(nn.Conv2d(1,64,kernel_size=(k_size_1d,H_neurons*frame_size*frame_size),
                                             padding = 0),act_fn())       
                  
        ######################### fully connected layer #########################
        self.dropout = nn.Dropout(p = dropoutRate)
        self.fc1 = nn.Sequential(nn.Linear(64*(n_frames-k_size_1d+1), d_layer),act_fn())
        self.fc2 = nn.Linear(d_layer, n_classes)
        self.FC = nn.Sequential(self.dropout,self.fc1,self.dropout,self.fc2)
   
                     
    def forward(self, x): 
        #x shape: [n_samples,n_channels,n_frames,32,32]   
        #set_trace()
        convnets = []
        convnet = 0
        n_frames = x.shape[2]
        for i in range(n_frames):
            convnet = x[:,:,i,:,:]    #[n_samples,n_channels,32,32]        
            #cnn
            convnet = self.cnn(convnet) #[n_samples, 128, 4, 4]
            convnets.append(convnet)            
        convnets = torch.stack(convnets) #[n_frames, n_samples, 128, 4, 4]
              
        convnets = convnets.permute(1,0,2,3,4) #[n_samples, n_frames, 128, 4, 4]  
        
        #Conv-1d_over_flames
        #reshape
        convnets = convnets.view(-1,1, n_frames,  
                                 convnets.size(-1)*convnets.size(-2)*convnets.size(-3))
                                # [samples,1(channels),n_frames,128*4*4]        
        convnets = self.conv8(convnets)  #[n_samples, 64, 8, 1]
        
                
        # flatten and FC
        convnets = convnets.view(convnets.shape[0],-1)
        convnets = self.FC(convnets)           
        return convnets   


class Net_base_LSTM(nn.Module): 
    """
        video size: [n_samples,n_channels,n_frames,32,32]
        image size: [batch_size, n_channles, 32, 32]
        
        
        :param n_channels: number of color channels
        :param n_frames: number of frames in the video
        :param kernel: the size of kernel, eg: 3 or 5 --> (3,3) or (5,5)
        :param act_fn: activation function, eg: nn.ReLU, nn.Sigmoid, nn.LeakyReLU
        :param num_units: number of LSTM units
        :param n_LSTM_layers: number of LSTM layers
        :param d_layer : the number of neurons in the last dense layer.
        :param dropoutRate : dropout rate.
                
    """   
    def __init__(self, act_fn,kernel,d_layer,n_channels,n_classes,dropoutRate,n_frames,num_units,n_LSTM_layers):               
        super(Net_base_LSTM, self).__init__()
        ######################### CNN #########################
        self.cnn, frame_size, H_neurons = CNN_model(kernel,act_fn,n_channels,dropoutRate)        
        #after the stack of frames, the size of data becames: [n_samples, 128, 4, 4,n_frames]  
        
        
        ######################### LSTM #########################
        self.lstm = nn.LSTM(input_size= H_neurons*frame_size*frame_size, hidden_size = num_units,num_layers = n_LSTM_layers)   
        
               
        ######################### fully connected layer #########################
        self.dropout = nn.Dropout(p = dropoutRate)
        self.fc1 = nn.Sequential(nn.Linear(num_units, d_layer),act_fn())
        self.fc2 = nn.Linear(d_layer, n_classes)
        self.FC = nn.Sequential(self.dropout,self.fc1,self.dropout,self.fc2)
        
     
    def forward(self, x): 
        #x shape: [n_samples,n_channels,n_frames,32,32]   
        #set_trace()
        convnets = []
        convnet = 0
        n_frames = x.shape[2]
        for i in range(n_frames):
            convnet = x[:,:,i,:,:]    #[n_samples,n_channels,32,32]        
            #cnn
            convnet = self.cnn(convnet) #[n_samples, 128, 6, 6]
            convnets.append(convnet)            
        convnets = torch.stack(convnets) #[n_frames, n_samples, 128, 6, 6]
        
        convnets = convnets.permute(1,0,2,3,4) #[n_samples, n_frames, 128, 6, 6]  
        
        #reshape
        convnets = convnets.view(-1, n_frames,  
                                 convnets.size(-1)*convnets.size(-2)*convnets.size(-3))
                                # [samples,n_frames,128*6*6]           
        #LSTM cell
        # input: (seq_len, batch, input_size)
        #h_0, c_0 are all defaut zero.
        convnets = convnets.permute(1,0,2) 
        output, (hn, cn) = self.lstm(convnets)  
        # output shape: (n_frames, samples, num_directions * hidden_size): Size([10, 32, 128])
        # hn shape: (num_layers * num_directions, samples, hidden_size), the hidden state for t = n_frames, [1, 32, 128]
        # c_n shape: (num_layers * num_directions, samples, hidden_size), tensor containing the cell state for t = n_frames, [1, 32, 128]
        #take the last output
        output = output[-1,:,:] #size([32,128])
                
                
        # flatten and FC
        output = self.FC(output)           
        return output  

class Net_base_Mix(nn.Module):
    """
        video size: [n_samples,n_channels,n_frames,32,32]
        image size: [batch_size, n_channles, 32, 32]
        
        
        :param n_channels: number of color channels
        :param n_frames: number of frames in the video
        :param kernel: the size of kernel, eg: 3 or 5 --> (3,3) or (5,5)
        :param act_fn: activation function, eg: nn.ReLU, nn.Sigmoid, nn.LeakyReLU
        :param num_units: number of LSTM units
        :param n_LSTM_layers: number of LSTM layers
        :param d_layer : the number of neurons in the last dense layer.
        :param k_size_1d: kernel size of 1d conv(in time domain)
        :param dropoutRate : dropout rate.
                
    """   


    def __init__(self, act_fn,kernel,d_layer,n_channels,n_classes,dropoutRate,n_frames,k_size_1d,num_units,n_LSTM_layers):
        super(Net_base_Mix, self).__init__()
        ######################### CNN #########################
        self.cnn, frame_size, H_neurons = CNN_model(kernel,act_fn,n_channels)        
        #after the stack of frames, the size of data becames: [n_samples, 128, 4, 4,n_frames]  

        ######################### LSTM #########################
        self.lstm = nn.LSTM(input_size= H_neurons*frame_size*frame_size, hidden_size = num_units,num_layers = n_LSTM_layers)
        
        ######################### conv1d #########################
        self.conv8 = nn.Sequential(nn.Conv2d(1,64,kernel_size=(k_size_1d,H_neurons*frame_size*frame_size),
                                             padding = 0),act_fn())         
        
        ######################### fully connected layer #########################
        self.dropout = nn.Dropout(p = dropoutRate)
        self.fc1 = nn.Sequential(nn.Linear(num_units+64*(n_frames-k_size_1d+1), d_layer),act_fn())
        self.fc2 = nn.Linear(d_layer, n_classes)
        self.FC = nn.Sequential(self.dropout,self.fc1,self.dropout,self.fc2)

        
    
    def forward(self, x): 
        #x shape: [n_samples,n_channels,n_frames,32,32]   
        #set_trace()
        convnets = []
        convnet = 0
        n_frames = x.shape[2]
        for i in range(n_frames):
            convnet = x[:,:,i,:,:]    #[n_samples,n_channels,32,32]        
            #cnn
            convnet = self.cnn(convnet) #[n_samples, 128, 6, 6]
            convnets.append(convnet)            
        convnets = torch.stack(convnets) #[n_frames, n_samples, 128, 6, 6]       
       
        convnets = convnets.permute(1,0,2,3,4) #[n_samples, n_frames, 128, 6, 6]  
        
        #LSTM cell
        #reshape 
        convnets_lstm = convnets.view(-1, n_frames,  
                                 convnets.size(-1)*convnets.size(-2)*convnets.size(-3))
                                # [samples,n_frames,128*6*6]           
        #LSTM cell
        # input: (seq_len, batch, input_size)
        #h_0, c_0 are all defaut zero.
        convnets_lstm = convnets_lstm.permute(1,0,2) 
        output, (hn, cn) = self.lstm(convnets_lstm)  
        # output shape: (n_frames, samples, num_directions * hidden_size): Size([10, 32, 128])
        # hn shape: (num_layers * num_directions, samples, hidden_size), the hidden state for t = n_frames, [1, 32, 128]
        # c_n shape: (num_layers * num_directions, samples, hidden_size), tensor containing the cell state for t = n_frames, [1, 32, 128]
        #take the last output
        output = output[-1,:,:] #size([32,128])
        
        #Conv-1d_over_flames
        #reshape
        convets_conv1d = convnets.view(-1,1, n_frames,  
                                 convnets.size(-1)*convnets.size(-2)*convnets.size(-3))
                                # [samples,1(channels),n_frames,128*6*6]   
        convets_conv1d = self.conv8(convets_conv1d)  #[n_samples, 64, 8, 1]        
        convets_conv1d = convets_conv1d.view(convets_conv1d.size(0),-1)
        
        #concat lstm and con1d
        concat = torch.cat((output,convets_conv1d),1)        
                
                
        # flatten and FC
        concat = self.FC(concat)           
        return concat  


class Net_base_CNN(nn.Module):    
    def __init__(self, act_fn,kernel,d_layer,n_channels,n_classes,dropoutRate):
        """
        image size: [batch_size, n_channles, 32, 32]
           
        :param n_channels: number of color channels
        :param n_frames: number of frames in the video
        :param kernel: the size of kernel, eg: 3 or 5 --> (3,3) or (5,5)
        :param act_fn: activation function, eg: nn.ReLU, nn.Sigmoid, nn.LeakyReLU
        :param d_layer : the number of neurons in the last dense layer.
        :param dropoutRate : dropout rate.
        """
        
        super(Net_base_CNN, self).__init__()
        ######################### CNN #########################
        self.cnn, frame_size, H_neurons = CNN_model(kernel,act_fn,n_channels,dropoutRate)        
        #size of the data: [n_samples, 128, 4, 4]
                       
        ######################### fully connected layer #########################
        self.dropout = nn.Dropout(p = dropoutRate)
        self.fc1 = nn.Sequential(nn.Linear(H_neurons*frame_size*frame_size, d_layer),act_fn())
        self.fc2 = nn.Linear(d_layer, n_classes)
        self.FC = nn.Sequential(self.dropout,self.fc1,self.dropout,self.fc2)

        
    def forward(self, x): 
        #x shape: [n_samples,n_channels,n_frames,32,32] 
        # average videos
        x = torch.mean(x,2) 
        convnet = self.cnn(x) #[n_samples, 128, 4, 4]
        
        # flatten and FC
        convnet = convnet.view(convnet.shape[0],-1)
        convnet = self.FC(convnet)           
        return convnet

class Net_base_FC(nn.Module):    
    def __init__(self, act_fn,d_layer,n_channels,n_classes,n_frames,dropoutRate):
        super(Net_base_FC, self).__init__()
       ######################### fully connected layer ######################### 
        self.dropout = nn.Dropout(p = dropoutRate)      
        self.fc1 = nn.Sequential(nn.Linear(n_channels*n_frames*32*32, d_layer),act_fn())
        self.fc2 = nn.Linear(d_layer, n_classes)
        self.FC = nn.Sequential(self.dropout,self.fc1,self.dropout,self.fc2)

    def forward(self, x): 
        #x shape: [n_samples,n_channels,n_frames,32,32] 
        # flatten and FC
        x = x.view(x.shape[0],-1)
        x = self.FC(x)           
        return x
    
    

# #####################################################################################
# 2 LAYERS VGG NET

def CNN_model_reduced(kernel, act_fn, n_channels, dropoutRate):
        ######################### CNN #########################
        ######################### layer 1 of cnn #########################
        #caculate the padding 'same'
        assert((kernel -1)%2  == 0)
        padding = int((kernel -1)/2)       
        conv1 = nn.Sequential(nn.Conv2d(n_channels, 32, kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn()) 
        frame_size = 32
        H_neurons = 32 #[batch_size, H_neurons, frame_size, frame_size]  
    
        conv2 = nn.Sequential(nn.Conv2d(32,32,kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn()) 
        frame_size = frame_size 
        H_neurons = 32 #[batch_size, H_neurons, frame_size, frame_size] 
                
        conv3 = nn.Sequential(nn.Conv2d(32,32,kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn()) 
        frame_size = frame_size 
        H_neurons = 32 #[batch_size, H_neurons, frame_size, frame_size] 
        
        conv4 = nn.Sequential(nn.Conv2d(32,32,kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn()) 
        frame_size = frame_size 
        H_neurons = 32 #[batch_size, H_neurons, frame_size, frame_size] 
                
        maxpool1 = nn.MaxPool2d(kernel_size = (2,2))
        frame_size = frame_size/2  #16
        H_neurons = 32 #[batch_size, H_neurons, frame_size, frame_size] 
                
        ######################### layer 2 of cnn #########################
        conv5 = nn.Sequential(nn.Conv2d(32,64,kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn()) 
        frame_size = frame_size  #16
        H_neurons = 64 #[batch_size, H_neurons, frame_size, frame_size] 
        
        conv6 = nn.Sequential(nn.Conv2d(64,64,kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn())  
        frame_size = frame_size  #16
        H_neurons = 64 #[batch_size, 64, 16, 16] 
        
        maxpool2 = nn.MaxPool2d(kernel_size = (2,2))
        assert(frame_size % 2  == 0)
        frame_size = frame_size//2 #8
        H_neurons = 64 #[batch_size, 64, 8, 8] 
        
        ######################### layer 3 of cnn #########################
        conv7 = nn.Sequential(nn.Conv2d(64,128,kernel_size=(kernel,kernel),padding = (padding,padding)),
                                   act_fn())   
        frame_size = frame_size #8
        H_neurons = 128 #[batch_size, 128, 8, 8] 
        
        maxpool3 = nn.MaxPool2d(kernel_size = (2,2)) 
        assert(frame_size % 2 == 0)
        frame_size = int(frame_size/2)  #4
        H_neurons = 128 #[batch_size, 128, 4, 4] 
                
        ### ---> integrate cnn layers <--- ####
        cnn = nn.Sequential(conv1,conv2, maxpool1,
                            conv5,conv6, maxpool2,
                            conv7, maxpool3 )
#         print(' - ONE MORE DROPOUT - ONE MORE DROPOUT - ONE MORE DROPOUT - ONE MORE DROPOUT - ONE MORE DROPOUT - ')
#         dropout = nn.Dropout(p = dropoutRate) 
#         cnn = nn.Sequential(conv1,conv2, maxpool1,
#                             conv5,conv6, maxpool2, dropout,
#                             conv7, maxpool3 )
 
        
        return cnn, int(frame_size), int(H_neurons)
    
    
class LSTM_with_reduced_CNN(nn.Module): 
    """
        video size: [n_samples,n_channels,n_frames,32,32]
        image size: [batch_size, n_channles, 32, 32]
        
        
        :param n_channels: number of color channels
        :param n_frames: number of frames in the video
        :param kernel: the size of kernel, eg: 3 or 5 --> (3,3) or (5,5)
        :param act_fn: activation function, eg: nn.ReLU, nn.Sigmoid, nn.LeakyReLU
        :param num_units: number of LSTM units
        :param n_LSTM_layers: number of LSTM layers
        :param d_layer : the number of neurons in the last dense layer.
        :param dropoutRate : dropout rate.
                
    """   
    def __init__(self, act_fn,kernel,d_layer,n_channels,n_classes,dropoutRate,n_frames,num_units,n_LSTM_layers):               
        super(LSTM_with_reduced_CNN, self).__init__()
        ######################### CNN #########################
        self.cnn, frame_size, H_neurons = CNN_model_reduced(kernel,act_fn,n_channels,dropoutRate)        
        #after the stack of frames, the size of data becames: [n_samples, 128, 4, 4,n_frames]  
        
        
        ######################### LSTM #########################
        self.lstm = nn.LSTM(input_size= H_neurons*frame_size*frame_size, hidden_size = num_units,num_layers = n_LSTM_layers)   
        
               
        ######################### fully connected layer #########################
        self.dropout = nn.Dropout(p = dropoutRate)
        self.fc1 = nn.Sequential(nn.Linear(num_units, d_layer),act_fn())
        self.fc2 = nn.Linear(d_layer, n_classes)
        self.FC = nn.Sequential(self.dropout,self.fc1,self.dropout,self.fc2)
        
     
    def forward(self, x): 
        #x shape: [n_samples,n_channels,n_frames,32,32]   
        #set_trace()
        convnets = []
        convnet = 0
        n_frames = x.shape[2]
        for i in range(n_frames):
            convnet = x[:,:,i,:,:]    #[n_samples,n_channels,32,32]        
            #cnn
            convnet = self.cnn(convnet) #[n_samples, 128, 6, 6]
            convnets.append(convnet)            
        convnets = torch.stack(convnets) #[n_frames, n_samples, 128, 6, 6]
        
        convnets = convnets.permute(1,0,2,3,4) #[n_samples, n_frames, 128, 6, 6]  
        
        #reshape
        convnets = convnets.view(-1, n_frames,  
                                 convnets.size(-1)*convnets.size(-2)*convnets.size(-3))
                                # [samples,n_frames,128*6*6]           
        #LSTM cell
        # input: (seq_len, batch, input_size)
        #h_0, c_0 are all defaut zero.
        convnets = convnets.permute(1,0,2) 
        output, (hn, cn) = self.lstm(convnets)  
        # output shape: (n_frames, samples, num_directions * hidden_size): Size([10, 32, 128])
        # hn shape: (num_layers * num_directions, samples, hidden_size), the hidden state for t = n_frames, [1, 32, 128]
        # c_n shape: (num_layers * num_directions, samples, hidden_size), tensor containing the cell state for t = n_frames, [1, 32, 128]
        #take the last output
        output = output[-1,:,:] #size([32,128])
                
                
        # flatten and FC
        output = self.FC(output)           
        return output  