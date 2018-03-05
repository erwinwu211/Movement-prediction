import numpy as np
import pandas as pd
import tflearn
import os
import matplotlib.pyplot as plt


def rmse(y_pred, y_true):
    return np.sqrt(((y_true - y_pred) ** 2).mean())

def rmsle(y_pred, y_true):
    return np.sqrt(np.square(np.log(y_true + 1) - np.log(y_pred + 1)).mean())

def mae(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred)))

def mape(y_pred, y_true):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

filepath='/home/erwinwu/Dataset/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'

dataset = np.empty([163,1], dtype='float32')
file_count=-1
for filename in sorted(os.listdir(filepath)):
	temp=False;
	file_count=file_count+1
	filename=filepath+'/'+filename
	degree = pd.read_table(filename,usecols=[1],engine='python').values
	number = pd.read_table(filename,usecols=[0],engine='python').values
	i=0
	for num in number:		
		if num == 126:
			temp=True;break;
		i=i+1
	if temp:
		dataset[file_count]= degree[i]
	else: 
		dataset[file_count]= 0
#normalize
dataset=dataset/3.14159266


#sine wave test dataset
dataset = np.empty([500,1], dtype='float32')
for x in range(500):
	dataset[x]=np.sin(x/30)

###original dataset
dataframe = pd.read_csv('1.csv',
        usecols=[1],
        engine='python',
        skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
dataset -= np.min(np.abs(dataset))
dataset /= np.max(np.abs(dataset))

##prepare data for training
def create_dataset(dataset, steps_of_history, steps_in_future):
    X, Y = [], []
    for i in range(0, len(dataset)-steps_of_history, steps_in_future):
        X.append(dataset[i:i+steps_of_history])
        Y.append(dataset[i + steps_of_history])
    X = np.reshape(np.array(X), [-1, steps_of_history, 1])
    Y = np.reshape(np.array(Y), [-1, 1])
    return X, Y

def split_data(x, y, test_size=0.1):
    pos = round(len(x) * (1 - test_size))
    trainX, trainY = x[:pos], y[:pos]
    testX, testY   = x[pos:], y[pos:]
    return trainX, trainY, testX, testY

steps_of_history = 1
steps_in_future = 1

X, Y = create_dataset(dataset, steps_of_history, steps_in_future)
trainX, trainY, testX, testY = split_data(X, Y, 0.33)

#define network
#net = tflearn.input_data(shape=[None, steps_of_history, 1])
#net = tflearn.lstm(net, n_units=6)
#net = tflearn.fully_connected(net, 1, activation='linear')
#net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
#        loss='mean_square')

net = tflearn.input_data(shape=[None, 1, 1])
tnorm = tflearn.initializations.uniform(minval=-1.0, maxval=1.0)
net = tflearn.lstm(net, 1, dropout=0.8)
net = tflearn.fully_connected(net, 1, activation='linear', weights_init=tnorm)
net = tflearn.regression(net, optimizer='adam', learning_rate=0.0001,loss='mean_square', metric='R2')

#start train
model = tflearn.DNN(net, tensorboard_verbose=0, clip_gradients=0)
#model.fit(trainX, trainY, validation_set=0.1, batch_size=1, n_epoch=100)
#model.save('my_model.tflearn')
# use  $ tensorboard --logdir=/tmp/tflearn_logs to check the logs

model.fit(trainX, trainY, n_epoch=100, batch_size=10, shuffle=False, show_metric=True)
score = model.evaluate(X, y, batch_size=128)
model.save('my_model.tflearn')





