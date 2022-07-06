###### import all necessary frameworks
import pandas as pd
import numpy as np
from pmdarima.arima import AutoARIMA
import plotly.express as px
import plotly.graph_objects as go
from tqdm.notebook import tqdm
from sklearn.metrics import mean_squared_error
from datetime import date, timedelta
import yfinance as yf

# Getting the date five years ago to download the current timeframe
years = (date.today() - timedelta(weeks=260)).strftime("%Y-%m-%d")

# Stocks to analyze
stocks = ['AAPL', 'TSLA', 'NVDA', 'MSFT']

# Getting the data for multiple stocks
df = yf.download(stocks, start=years).dropna()

# Storing the dataframes in a dictionary
stock_df = {}

for col in set(df.columns.get_level_values(0)):

    # Assigning the data for each stock in the dictionary
    stock_df[col] = df[col]

# Finding the log returns
stock_df['LogReturns'] = stock_df['Adj Close'].apply(np.log).diff().dropna()

# Using Moving averages
stock_df['MovAvg'] = stock_df['Adj Close'].rolling(10).mean().dropna()

# Logarithmic scaling of the data and rounding the result
stock_df['Log'] = stock_df['MovAvg'].apply(np.log).apply(lambda x: round(x, 2))

# Days in the past to train on
days_to_train = 180

# Days in the future to predict
days_to_predict = 5

# Establishing a new DF for predictions
stock_df['Predictions'] = pd.DataFrame(index=stock_df['Log'].index,
                                       columns=stock_df['Log'].columns)

# Iterate through each stock
for stock in tqdm(stocks):

    # Current predicted value
    pred_val = 0

    # Training the model in a predetermined date range
    for day in tqdm(range(1000,
                          stock_df['Log'].shape[0]-days_to_predict)):

        # Data to use, containing a specific amount of days
        training = stock_df['Log'][stock].iloc[day-days_to_train:day+1].dropna()

        # Determining if the actual value crossed the predicted value
        cross = ((training[-1] >= pred_val >= training[-2]) or
                 (training[-1] <= pred_val <= training[-2]))

        # Running the model when the latest training value crosses the predicted value or every other day
        if cross or day % 2 == 0:

            # Finding the best parameters
            model    = AutoARIMA(start_p=0, start_q=0,
                                 start_P=0, start_Q=0,
                                 max_p=8, max_q=8,
                                 max_P=5, max_Q=5,
                                 error_action='ignore',
                                 information_criterion='bic',
                                 suppress_warnings=True)

            # Getting predictions for the optimum parameters by fitting to the training set
            forecast = model.fit_predict(training,
                                         n_periods=days_to_predict)

            # Getting the last predicted value from the next N days
            stock_df['Predictions'][stock].iloc[day:day+days_to_predict] = np.exp(forecast[-1])


            # Updating the current predicted value
            pred_val = forecast[-1]

# Shift ahead by 1 to compare the actual values to the predictions
pred_df = stock_df['Predictions'].shift(1).astype(float).dropna()

for stock in stocks:

    fig = go.Figure()

    # Plotting the actual values
    fig.add_trace(go.Scatter(x=pred_df.index,
                             y=stock_df['MovAvg'][stock].loc[pred_df.index],
                             name='Actual Moving Average',
                             mode='lines'))

    # Plotting the predicted values
    fig.add_trace(go.Scatter(x=pred_df.index,
                             y=pred_df[stock],
                             name='Predicted Moving Average',
                             mode='lines'))

    # Setting the labels
    fig.update_layout(title=f'Predicting the Moving Average for the Next {days_to_predict} days for {stock}',
                      xaxis_title='Date',
                      yaxis_title='Prices')

    fig.show()
