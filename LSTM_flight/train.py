import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

from create_data import dataset
from create_data import split_dataset

BATCHSIZE=1
EPOCH=200
look_back = 12

datapath='/home/erwinwu/Dataset/ApplyEyeMakeup/'
dataset1 = dataset('Data/Soccer')
#filepath='/home/wt/Dataset/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'


def fit_model(trainX,trainY,BATCH_SIZE,look_back):
	model = Sequential()
	model.add(LSTM(4, input_shape=(trainX.shape[1],look_back),stateful=False))	
	#model.add(keras.layers.wrappers.TimeDistributed(Dense(look_back)))
	model.add(Dense(1))	
	model.compile(loss='mean_squared_error', optimizer='adam')
	for i in range(EPOCH):
		model.fit(trainX, trainY,\
			 batch_size=BATCH_SIZE,\
			 epochs= 1,\
			 verbose=2,\
			 shuffle=False)
		print("Epochs:"+str(i))
		model.reset_states()
	return model

#for filename in sorted(os.listdir(datapath)):
	#filepath = datapath+filename
	#print(filepath)

print(dataset1)
trainX,trainY,testX,testY = split_dataset(dataset1,look_back)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

model = fit_model(trainX, trainY,BATCHSIZE,look_back)

model.save("model.h5")





