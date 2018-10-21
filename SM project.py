import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
#Part-1
#importing training set
dataset_train = pd.read_csv("data till sept.csv")
dataset_train['Shares Traded'] = dataset_train['Shares Traded'].fillna((dataset_train['Shares Traded'].mean()))
training_set = dataset_train.iloc[:,1:6].values


#feature scaling using normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
trainingset_scaled = sc.fit_transform(training_set)

sc_predict = MinMaxScaler()
trainingset_scaled_predict=sc_predict.fit_transform(training_set[:,0:1])
 
#creating a datstructure with 50 timesteps and 1 output.
#50 timesteps means that the at any time t the rnn with look at the trend from 50 days back
x_train=[]
y_train=[]
for i in range(50,5411):
    x_train.append(trainingset_scaled[i-50:i,:])
    y_train.append(trainingset_scaled[i,0])
x_train,y_train = np.array(x_train),np.array(y_train)  

#reshaping. to add another dimension ie the indicator/s
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],5))

#Part-2 Building the rnn

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout 
#initializing the rnn
regressor = Sequential()

#adding the 1st lstm layer and some dropout regularization 
#units refers to number of neurons in the hidden layer
#neurons are 50 to maintain high dimentionality
regressor.add(LSTM(units = 100,return_sequences = True, input_shape=(x_train.shape[1],5)))
regressor.add(Dropout(0.2))

#adding the 2st lstm layer and some dropout regularization. 
#No need to specify input_shape since it is automatically recognized
regressor.add(LSTM(units = 100,return_sequences = True))
regressor.add(Dropout(0.2))

#adding the 3st lstm layer and some dropout regularization
regressor.add(LSTM(units = 100,return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 100,return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 100,return_sequences = True))
regressor.add(Dropout(0.2))
 
#adding the 4st lstm layer and some dropout regularization 
#return_sequences  false. no need to return any sequences for the last layer
regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.2))

#adding the output layer
regressor.add(Dense(units = 1))

#Compiling the rnn.
#The loss function will mean squared error since this a regression problem
regressor.compile(optimizer = 'adam',loss="mean_squared_error")

#fitting the rnn to the training set
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
tb = TensorBoard('logs')
regressor.fit(x_train,y_train,epochs=10,callbacks=[es, rlr, tb], batch_size =32)

#Part-3 Making the predictions and visualising the results
dataset_test = pd.read_csv("test_set.csv")
real_stock_price = dataset_test.iloc[:,1:2].values
#getting the predicted stock prices
#for vertical concatenation use axis =0
parameters=['Open','High','Low','Close','Shares Traded']
dataset_total = pd.concat((dataset_train[parameters],dataset_test[parameters]),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-50:].values
inputs=inputs.reshape(-1,5)
inputs = sc.transform(inputs)
x_test=[]
for i in range(50,68):
    x_test.append(inputs[i-50:i,:])
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],5))
predicted_stock_price=regressor.predict(x_test)
#inverse the scale to get the actual prices 
from numpy import concatenate
x_test1 = x_test[:,0,:]
inv_pst = concatenate((predicted_stock_price, x_test1[:, 1:]), axis=1)
predicted_stock_price = sc.inverse_transform(inv_pst)
predicted_stock_price = predicted_stock_price[:,0]
plt.plot(real_stock_price,color = 'red',label ='Real stock price')
plt.plot(predicted_stock_price,color = 'blue',label ='Predicted stock price')
plt.title("Nifty50 Stock price prediction")
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()



