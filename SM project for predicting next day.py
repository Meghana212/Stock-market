import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import quandl
from pandas import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 
from sklearn.preprocessing import MinMaxScaler
#from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
#Part-1
#importing training set
dataset_train0 = pd.read_csv("training_set_1.csv")

last_date = dataset_train0.iloc[-1,[0]].values
last_date = str(last_date)
last_date=last_date[2:len(last_date)-2]
print(last_date)
#dataset_train1=dataset_train0.iloc[:,0:-1]

mydata = quandl.get("NSE/NIFTY_50",start_date=last_date)
#mydata['Index'] = range(0, len(mydata))
mydata=mydata.reset_index()
#mydata1 = DataFrame(data=mydata)
#mydata.rename(index=str,columns={'Index':'Date')
mydata1 = mydata.iloc[0:-1,:]

parameters=['Date','Open','High','Low','Close','Shares Traded','Turnover (Rs. Cr)']
dataset_train = pd.concat((dataset_train0[parameters],mydata[parameters]),axis=0)
dataset_train = dataset_train.reset_index()
#dataset_train = dataset_train.drop([index])
dataset_train['Close'] = dataset_train['Close'].fillna((dataset_train['Close'].mean()))

dataset_test = mydata.iloc[-1:,:]
training_set = dataset_train.iloc[:,[2,5]].values


#feature scaling using normalization

sc = MinMaxScaler()
trainingset_scaled = sc.fit_transform(training_set)

sc_predict = MinMaxScaler()
trainingset_scaled_predict=sc_predict.fit_transform(training_set[:,0:1])
 
#creating a datstructure with 50 timesteps and 1 output.
#50 timesteps means that the at any time t the rnn with look at the trend from 50 days back
x_train=[]
y_train=[]
for i in range(60,len(dataset_train)):
    x_train.append(trainingset_scaled[i-60:i,:])
    y_train.append(trainingset_scaled[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)  

#reshaping. to add another dimension ie the indicator/s
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],2))

#Part-2 Building the rnn

#initializing the rnn
regressor = Sequential()

#adding the 1st lstm layer and some dropout regularization 
#units refers to number of neurons in the hidden layer
#neurons are 50 to maintain high dimentionality
regressor.add(LSTM(units = 128,return_sequences = True, input_shape=(x_train.shape[1],5)))
regressor.add(Dropout(0.2))

#adding the 2st lstm layer and some dropout regularization. 
#No need to specify input_shape since it is automatically recognized
regressor.add(LSTM(units = 128,return_sequences = True))
regressor.add(Dropout(0.2))

#adding the 3st lstm layer and some dropout regularization

#adding the 4st lstm layer and some dropout regularization 
#return_sequences  false. no need to return any sequences for the last layer
regressor.add(LSTM(units = 128))
regressor.add(Dropout(0.2))
#adding the output layer
regressor.add(Dense(units = 1))

#Compiling the rnn.
#The loss function will mean squared error since this a regression problem
regressor.compile(optimizer = 'adam',loss="mean_squared_error")

#fitting the rnn to the training set
regressor.fit(x_train,y_train,epochs=180, batch_size =32)

#Part-3 Making the predictions and visualising the results
dataset_test = pd.read_csv("test_set_1.csv")
#real_stock_price = dataset_test.iloc[:,1:2].values
#getting the predicted stock prices
#for vertical concatenation use axis =0
parameters=['Open','Close']
dataset_total = pd.concat((dataset_train[parameters],dataset_test[parameters]),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,2)
inputs = sc.transform(inputs)
x_test=[]
for i in range(60,61):
    x_test.append(inputs[i-60:i,:])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],2))
predicted_stock_price=regressor.predict(x_test)
#inverse the scale to get the actual prices 
#from numpy import concatenate
#x_test1 = x_test[:,0,:]
#inv_pst = concatenate((predicted_stock_price, x_test1[:, 1:]), axis=1)
predicted_stock_price = sc_predict.inverse_transform(predicted_stock_price)
#predicted_stock_price = predicted_stock_price[:,0]





