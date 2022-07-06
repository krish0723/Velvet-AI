#IMPORT

# data libraries
import pandas as pd
import numpy as np

# Machine learning libraries
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Technical indicator library
import talib as ta

# Data import library
import yfinance as yf

#Data visualisation
import plotly.graph_objs as go

#Live/Paper Trading library
import alpaca_trade_api as tradeapi
# API Info for fetching data, portfolio, etc. from Alpaca
BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_API_KEY = "PKZIMKKU7IGAT5LRIE0Q"
ALPACA_SECRET_KEY = "i84pTqnlmf5QKEi63CcoHXovapFvCIRBylwScfDi"

# Instantiate REST API Connection
api = tradeapi.REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY,
                    base_url=BASE_URL, api_version='v2')

#DATA

#Pull relevant stock data
df = yf.download('AAPL',period = '1d', interval = '1m')
df

#Clean stock data (because of short time periods volume might be 0)
df = df.drop(df[df['Volume'] == 0].index)

# start adding financial prediction indicators
# build RSI calculation
n = 14
df['RSI'] = ta.RSI(np.array(df['Close'].shift(1)), timeperiod=n)
df

# build SMA calculation
df['SMA'] = df['Close'].shift(1).rolling(window=n).mean()
df

# build Correlation Coeffecient calculation
df['Corr'] = df['Close'].shift(1).rolling(window=n).corr(df['SMA'].shift(1))
df

# build SAR calculation
df['SAR'] = ta.SAR(np.array(df['High'].shift(1)), np.array(df['Low'].shift(1)), 0.2, 0.2)

# build ADX calculation
df['ADX'] = ta.ADX(np.array(df['High'].shift(1)), np.array(df['Low'].shift(1)), np.array(df['Open']), timeperiod=n)
df

# Create columns high, low and close with previous minute's OHLC data
df['Prev_High'] = df['High'].shift(1)
df['Prev_Low'] = df['Low'].shift(1)
df['Prev_Close'] = df['Close'].shift(1)

# Create columns 'OO' with the difference between the current minute's open and last minute's open
df['OO'] = df['Open']-df['Open'].shift(1)

# Create columns 'OC' with the difference between the current minute's open and last minute's close
df['OC'] = df['Open']-df['Prev_Close']

# Create a column 'Ret' with the calculation of returns
df['Ret'] = (df['Open'].shift(-1)-df['Open'])/df['Open']

# Create n columns and assign
for i in range(1, n):
    df['return%i' % i] = df['Ret'].shift(i)

# Change the value of 'Corr' to -1 if it is less than -1
df.loc[df['Corr'] < -1, 'Corr'] = -1


# Change the value of 'Corr' to 1 if it is greater than 1
df.loc[df['Corr'] > 1, 'Corr'] = 1

df = df.dropna()

t = .8
split = int(t*len(df))
split

import warnings
warnings.filterwarnings("ignore")

# Create a column by name, 'Signal' and initialize with 0
df['Signal'] = 0

# Assign a value of 1 to 'Signal' column for the quantile with the highest returns
df.loc[df['Ret'] > df['Ret'][:split].quantile(q=0.66), 'Signal'] = 1

# Assign a value of -1 to 'Signal' column for the quantile with the lowest returns
df.loc[df['Ret'] < df['Ret'][:split].quantile(q=0.34), 'Signal'] = -1

#TRAIN

# Use drop method to drop the columns and build training data set
X = df.drop(['Close', 'Signal', 'High', 'Low', 'Volume', 'Ret'], axis=1)

# Create a variable which contains all the 'Signal' values
y = df['Signal']

# Set hyperparameters to Test
c = [10, 100, 1000, 10000] # c val
g = [1e-2, 1e-1, 1e0] # gamma val, kernal val is hardcoded to rbf

# Intialise the parameters
parameters = {'svc__C': c,
              'svc__gamma': g,
              'svc__kernel': ['rbf']
              }

# Create the 'steps' variable with the pipeline functions
steps = [('scaler', StandardScaler()), ('svc', SVC())]

# Pass the 'steps' to the Pipeline function
pipeline = Pipeline(steps)

# Creating a randomized function to help to find the best parameters.
# Call the RandomizedSearchCV function and pass the parameters
rcv = RandomizedSearchCV(pipeline, parameters, cv=TimeSeriesSplit(n_splits=2))

# Call the 'fit' method of rcv and pass the train data to it
rcv.fit(X.iloc[:split], y.iloc[:split])

# Call the 'best_params_' method to obtain the best parameters of C
best_C = rcv.best_params_['svc__C']

# Call the 'best_params_' method to obtain the best parameters of kernel
best_kernel = rcv.best_params_['svc__kernel']

# Call the 'best_params_' method to obtain the best parameters of gamma
best_gamma = rcv.best_params_['svc__gamma']

print(best_C, best_kernel, best_gamma)

# Create a new SVC classifier
cls = SVC(C=best_C, kernel=best_kernel, gamma=best_gamma)

# Instantiate the StandardScaler
ss1 = StandardScaler()

# Pass the scaled train data to the SVC classifier
cls.fit(ss1.fit_transform(X.iloc[:split]), y.iloc[:split])

# Pass the test data to the predict function and store the values into 'y_predict'
y_predict = cls.predict(ss1.transform(X.iloc[split:]))

# Initiate a column by name, 'Pred_Signal' and assign 0 to it
df['Pred_Signal'] = 0

# Save the predicted values for the train data
df.iloc[:split, df.columns.get_loc('Pred_Signal')] = pd.Series(
    cls.predict(ss1.transform(X.iloc[:split])).tolist())

# Save the predicted values for the test data
df.iloc[split:, df.columns.get_loc('Pred_Signal')] = y_predict

# Calculate strategy returns and store them in 'Ret1' column
df['Ret1'] = df['Ret']*df['Pred_Signal']


# Calculate the confusion matrix
cm = confusion_matrix(y[split:], y_predict)
cm

# Calculate the classification report
cr = classification_report(y[split:], y_predict)
print(cr)

#declare figure
fig = go.Figure()

#Set up traces
fig.add_trace(go.Scatter(x=df.index, y= (df['Ret'][split:]+1).cumprod(),line=dict(color='royalblue', width=.8), name = 'stock_returns'))
fig.add_trace(go.Scatter(x=df.index, y= (df['Ret1'][split:]+1).cumprod(),line=dict(color='orange', width=.8), name = 'strategy_returns'))

# Add titles
fig.update_layout(
    title='Support Vector Machine Strategy',
    yaxis_title='Stock return (% Return)')

fig.show()
