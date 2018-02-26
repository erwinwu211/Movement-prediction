
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tflearn
from matplotlib import pyplot
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


filepath='/home/erwinwu/Dataset/ApplyEyeMakeup'

frame_count=0
datasetX=np.array([[0.0]]*192)  #initialize dataset

for foldername in os.listdir(filepath):
	for filename in sorted(os.listdir(filepath+'/'+foldername)):
		filename=filepath+'/'+foldername+'/'+filename  #absolute filepath
		print(filename)
		number = pd.read_table(filename,usecols=[0],engine='python').values
		degree = pd.read_table(filename,usecols=[1],engine='python').values
		distance = pd.read_table(filename,usecols=[2],engine='python').values
		#dataset normalize
		degree = degree.astype('float32')/3.14159266
		i=0
		for num in number:
		    datasetX[int(num)-1][frame_count]=float(degree[i])
		    i=i+1
		frame_count=frame_count+1
		datasetX=np.insert(datasetX,frame_count,0,axis=1)

		#datasetY = distance.astype('float32'
		#23datasetY -= np.min(np.abs(datasetY))
		#datasetY /= np.max(np.abs(datasetY))
datasetX=datasetX[:,:frame_count] #delete last column

def create_dataset(dataset, steps_of_history, steps_in_future):
    width = int (dataset.size / len(datasetX))
    X = dataset[:,:width-1]
    Y = dataset[:,1:]
    return X, Y

def split_data(x, y, test_size=0.1):
    pos = round(x.size/len(x) * (1 - test_size))
    trainX, trainY = x[:,:pos].transpose(), y[:,:pos].transpose()
    testX, testY   = x[:,pos:].transpose(), y[:,pos:].transpose()
    trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
    testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))
    trainY=trainY[:,98]
    testY=testY[:,98]
    return trainX, trainY, testX, testY

steps_of_history = 1
steps_in_future = 1

X, Y = create_dataset(datasetX, steps_of_history, steps_in_future)
trainX, trainY, testX, testY = split_data(X, Y, 0.33)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(trainX, trainY, epochs=50, batch_size=72, validation_data=(testX, testY), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()





#net = tflearn.input_data(1, 192)
#net = tflearn.lstm(net, n_units=6)
#net = tflearn.fully_connected(net, 1, activation='linear')
#net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
#        loss='mean_square')


#model = tflearn.DNN(net, tensorboard_verbose=1)
#model.fit(trainX, trainY, validation_set=0.1, batch_size=1, n_epoch=150)
#model.save('my_model.tflearn')
