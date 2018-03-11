import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

filepath='/home/erwinwu/Dataset/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
pred_step = 5
lattice_point = 5

def is_empty(any_structure):
    if any_structure:
        return False
    else:
        return True

def split_dataset(dataset,look_back,split_rate=1):
	train_size = int(len(dataset) * split_rate)
	train= dataset[0:train_size,:]
	trainX, trainY = create_dataset(train, look_back)
	if (split_rate != 1) :	
		test_size = len(dataset) - train_size	
		test = dataset[train_size:len(dataset),:]
		testX, testY = create_dataset(test, look_back)
	else:
		testX=trainX
		testY=trainY
	return trainX,trainY,testX,testY




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

def dataset(datapath=filepath):
	file_count=-1
	files_len=len(os.listdir(datapath))
	dataset = np.empty([files_len,lattice_point+1 ], dtype='float32')
	for filename in sorted(os.listdir(datapath)):
		file_count=file_count+1
		filename=datapath+'/'+filename
		#if file_count< 50: break;
		if os.path.getsize(filename) == 0:
			degree = np.zeros(768)
			number = np.zeros(768)
		else:
			degree = pd.read_table(filename,usecols=[1],engine='python').values
			number = pd.read_table(filename,usecols=[0],engine='python').values
		for i in range(470,470+lattice_point):
			if not is_empty(np.where(number==i+1)[0]):	#if number i+1 exists in the data
				num = int(np.where(number==i+1)[0])   	#then num is index of number i+1
				dataset[file_count,i-469]= degree[num]	
			else: 
				dataset[file_count,i-469]= 0
	
	#make prediction
	for y in range(files_len):
		if (y<(files_len - pred_step)):
			dataset[y,0]=dataset[y+pred_step,1]
		else:
			dataset[y,0]=0
	return dataset

