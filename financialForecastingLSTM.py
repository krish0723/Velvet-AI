import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

#get the stock quote
df = web.DataReader('LNVGY', data_source='yahoo', start='2012-01-01', end='2021-09-21')
#show the data

print("Initial Data Set Used: 2012-DOI")
df

#get the number of rows and columns in dataset
df.shape
#visualize the closing price history
plt.figure(figsize=(16,8))
plt.title("Close Price History")
plt.plot(df["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD ($)", fontsize=18)
print("Close Price History Graph:")
plt.show()

#Create a new dataframe with only Close column
data = df.filter(["Close"])
# convert the dataframe to a numpy array
dataset = data.values
#get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * .8)
training_data_len

# preprocessing
# scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print("Scaled Data Values: ")
scaled_data


# create the training data set
# create the scaled training data set
train_data = scaled_data[0:training_data_len, :]
# split the data into x_train and y_train
x_train = []
y_train = []
print("x_train and y_train values for each pass:")
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
    if (i <= 60):
        print(x_train)
        print(y_train)
        print()

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
print("Building and Compiling the Model, Training:")
model.fit(x_train, y_train, batch_size=1, epochs=1)

# create the testing data set
# create a new array containing scaled values from index 1543 to 2003
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
print("Prediction Accuracy: 0 = Perfect Accuracy:")
rmse

#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

valid

#visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
print("Model Prediction vs Testing Data, Time Series Graph:")
plt.show()

#show the valid and predicted prices
print("\n Testing Data vs Prediction Data: ")
valid
valid_percentChange = ((valid['Predictions'] - valid['Close'])/valid['Close'])*100
print("Percent Deviation: ")
valid_percentChange
#get the quote
apple_quote = web.DataReader('PLTR', data_source='yahoo', start='2012-01-01', end='2021-09-29')
#create a new dataframe
new_df = apple_quote.filter(['Close'])
# get the last 60 day closing price values and convert the dataframe to an array
last_60_days = new_df[-60:].values
#scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
#create an empty list
X_test = []
# append the last 60 days to X_test
X_test.append(last_60_days_scaled)
# convert the X_test data set to a numpy array
X_test = np.array(X_test)
#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# get the predicted scaled price
pred_price = model.predict(X_test)
# undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print("Predicted Price on Entered Date: ")
print(pred_price)

#get the quote
apple_quote2 = web.DataReader('PLTR', data_source='yahoo', start='2021-09-22', end='2021-09-28')
print("Actual Close Price: ")
print(apple_quote2['Close'])
