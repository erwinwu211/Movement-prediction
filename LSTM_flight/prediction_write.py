import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import cv2
import math
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

# make predictions
trainPredict = model.predict(trainX,1)

cap = cv2.VideoCapture("soccer1.mp4")
cv2.namedWindow("frame")

for i in trainPredict:	
	ret,frame = cap.read()
	c= (471//24)*20
	d= (471%24 -1)*20
	a= c+ int(20*math.cos(i))
	b= d+ int(20*math.sin(i))
	print (a,b,c,d)
	frame=cv2.arrowedLine(frame,(c,d),(a,b),[0,0,255],1)
	cv2.imshow('frame',frame)
	print(i)
	k = cv2.waitKey(100) & 0xff

