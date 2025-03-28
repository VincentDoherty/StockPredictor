import yfinance as yf
import pandas as pd

def backtest_strategy(stock_symbol, start_date, end_date, strategy):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data['Signal'] = strategy(data)
    data['Returns'] = data['Close'].pct_change()
    data['Strategy Returns'] = data['Returns'] * data['Signal'].shift(1)
    cumulative_returns = (1 + data['Strategy Returns']).cumprod() - 1
    return cumulative_returns

def example_strategy(data):
    data['Signal'] = 0
    data.loc[data['Close'] > data['Close'].rolling(window=20).mean(), 'Signal'] = 1
    data.loc[data['Close'] < data['Close'].rolling(window=20).mean(), 'Signal'] = -1
    return data['Signal']