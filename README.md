# Stock-market-prediction
The aim of the project is to predict stock market using machine learning. It is
related to ML as it tries to predict the trend of stock prices by doing technical
analysis.
The project focuses on the Indian stock market, more specifically NIFTY 50.
The objective is achieved using a type o neural network called RNN(Recurrent
Neural Network) more specifically LSTM&#39;s (Long Short Term Memory).
LSTM&#39;s units are memory units of recurrent neural network. A common LSTM
unit is composed of a cell, an input gate, an output gate and a forget gate.
It takes historical data from quandl which consists of open prices for NIFTY 50.
These values make up the training,validation and test set.
The neural network is trained on the training set of a period of 60 days to predict
the stock price of the 61 st day. This keeps repeating till the end of the training is
reached. The weights for the parameter are adjusted to make a model which can
predict the stock price of next day accurately.
The neural network predicts the prices for the period in the test set and the
predicted values are compared to the values in test set and the weights are adjusted
accordingly.
