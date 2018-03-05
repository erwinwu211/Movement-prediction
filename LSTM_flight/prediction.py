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


# reshape into X=t and Y=t+1
look_back = 12
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


model = keras.models.load_model('model.keras')




# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
pad_col = np.zeros(dataset.shape[1]-1)
# invert predictions
def pad_array(val):
    return np.array([np.insert(pad_col, 0, x) for x in val])
    
#trainPredict = scaler.inverse_transform(pad_array(trainPredict))
#trainY = scaler.inverse_transform(pad_array(trainY))
#testPredict = scaler.inverse_transform(pad_array(testPredict))
#testY = scaler.inverse_transform(pad_array(testY))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions


plt.plot(dataset[:,1],label="input")
plt.plot(trainPredict,label="test_out")
plt.plot(testPredictPlot[:,0],label="realtime_out")
plt.legend()
plt.title('Prediction')
plt.savefig('figure1.png')
