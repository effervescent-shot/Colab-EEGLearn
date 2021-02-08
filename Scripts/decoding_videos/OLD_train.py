### OLD APPROACH: TRAINING OF 3D CONV

from Conv_3d import *

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

#train(100,loader_train,loader_test)
train_tune(100,loader_train,loader_test)

