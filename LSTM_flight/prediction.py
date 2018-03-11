import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from create_data import dataset
from create_data import split_dataset

# convert an array of values into a dataset matrix
# if you give look_back 3, a part of the array will be like this: Jan, Feb, Mar
dataset = dataset('Data/Soccer')
look_back = 12


def pad_array(val):
    return np.array([np.insert(pad_col, 0, x) for x in val])

model = keras.models.load_model('model.h5')
# split into train and test sets

trainX,trainY,testX,testY = split_dataset(dataset,look_back)

print (testX,testY)
# make predictions
trainPredict = model.predict(trainX,1)
#testPredict = model.predict(testX)
pad_col = np.zeros(dataset.shape[1]-1)
# invert predictions

#scale back
#trainPredict = scaler.inverse_transform(pad_array(trainPredict))
#trainY = scaler.inverse_transform(pad_array(trainY))
#testPredict = scaler.inverse_transform(pad_array(testPredict))
#testY = scaler.inverse_transform(pad_array(testY))

# shift test predictions for plotting
#testPredictPlot = np.empty_like(dataset)
#testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(dataset[:,1],label="input")
plt.xlim(0,200)
plt.legend()
plt.title('Iutput')
plt.savefig('figure1.png')
plt.clf()
plt.plot(trainPredict,label="test_out",color='red')
#plt.plot(testPredictPlot[:,0],label="realtime_out",color='red')
plt.xlim(0,200)
plt.legend()
plt.title('Output')
plt.savefig('figure2.png')
