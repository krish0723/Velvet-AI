import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

class ModelLSTM:

    def __init__(self):
        self.df = web.DataReader('LNVGY', data_source='yahoo', start='2012-01-01', end='2021-09-21')
        df.shape
        data = df.filter(["Close"])
        self.dataset = data.values
        self.training_data_len = math.ceil(len(dataset) * .8)

    def createModel(self):
        # preprocessing
        # scale the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(this.dataset)

        # create the training data set
        # create the scaled training data set
        train_data = scaled_data[0:this.training_data_len, :]

        # split the data into x_train and y_train
        x_train = []
        y_train = []
        """print("x_train and y_train values for each pass:")
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
            if (i <= 60):
                print(x_train)
                print(y_train)
                print()"""

        # convert the x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        # reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_train.shape

        # built the lstm model
        model = Sequential()
        model.add(LSTM(500, return_sequences=True, input_shape = (x_train.shape[1], 1)))
        model.add(LSTM(500, return_sequences=False))
        model.add(Dense(250))
        model.add(Dense(1))

        # compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        #train the model
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        return model, scaled_data

    def percentError(self, model, scaled_data):
        # create the testing data set
        # create a new array containing scaled values from index 1543 to 2003
        scaled_data = scaled_data
        test_data = scaled_data[training_data_len - 60: , :]
        #create the data sets x_test and y_test
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        #convert the data to a numpy array
        x_test = np.array(x_test)

        #reshape the data
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        #get the models predicted price values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        #get the root mean squared error (RMSE) ! 0 = perfect prediction
        rmse = np.sqrt( np.mean ( predictions - y_test)**2 )

        return rmse
