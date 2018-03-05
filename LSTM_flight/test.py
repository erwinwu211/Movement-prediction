import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import keras
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#load data, the first col is the prediction

#dataframe = pandas.read_csv('1.csv', usecols=[0,1], engine='python', skipfooter=1)
#print(dataframe.head())
#dataset = dataframe.values
#dataset = dataset.astype('float32')

dataset = np.empty([500,2], dtype='float32')
for x in range(500):
	dataset[x,1]=np.sin(x/30)

for x in range(500):
	if (x<495):
		dataset[x,0]=dataset[x+5,1]
	else:
		dataset[x,0]=dataset[499,1]

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))


# convert an array of values into a dataset matrix
# if you give look_back 3, a part of the array will be like this: Jan, Feb, Mar
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        xset = []
        for j in range(dataset.shape[1]):
            a = dataset[i:(i+look_back), j]
            xset.append(a)
        dataY.append(dataset[i + look_back, 0])      
        dataX.append(xset)
    return np.array(dataX), np.array(dataY)

# reshape into X=t and Y=t+1
look_back = 12
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(testX.shape)
print(testX[0])
print(testY)

# reshape input to be [samples, time steps(number of variables), features] *convert time series into column
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))


# create and fit the LSTM network

###
old_session = KTF.get_session()

session = tf.Session('')
KTF.set_session(session)
KTF.set_learning_phase(1)
###

model = Sequential()
model.add(LSTM(4, input_shape=(testX.shape[1], look_back)))	
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
tb_cb = keras.callbacks.TensorBoard(log_dir='~/tflog/', histogram_freq=1)
cbks = [tb_cb]
history = model.fit(trainX, trainY, batch_size=2, epochs= 1000, verbose=2, callbacks=cbks,  validation_data=(testX, testY))
model.save("model.keras")

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
pad_col = np.zeros(dataset.shape[1]-1)

# invert predictions
def pad_array(val):
    return np.array([np.insert(pad_col, 0, x) for x in val])
    
trainPredict = scaler.inverse_transform(pad_array(trainPredict))
trainY = scaler.inverse_transform(pad_array(trainY))
testPredict = scaler.inverse_transform(pad_array(testPredict))
testY = scaler.inverse_transform(pad_array(testY))

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

print(testY[:,0])
print(testPredict[:,0])
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.savefig('figure.png')
