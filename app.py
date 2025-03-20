import yfinance as yf
import plotly.express as px
from flask import Flask, render_template_string, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def plot():
    stock_symbol = 'GOOGL'  # Default stock symbol
    if request.method == 'POST':
        stock_symbol = request.form['stock_symbol']

    # Fetch data for the specified stock
    data = yf.download(stock_symbol, start='2020-01-01', end='2025-02-19')

    # Prepare the data for the LSTM model
    data = data[['Close']]
    data.columns = ['Close']  # Ensure the column name is 'Close'
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create the training data
    train_data = []
    target_data = []
    for i in range(1000, len(scaled_data) - 30):
        train_data.append(scaled_data[i-1000:i, 0])
        target_data.append(scaled_data[i:i+30, 0])
    train_data, target_data = np.array(train_data), np.array(target_data)

    # Reshape the data for the LSTM model
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(train_data.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(30))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(train_data, target_data, epochs=10, batch_size=32)

    # Make predictions for the next 30 days
    last_60_days = scaled_data[-60:]
    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
    predicted_prices = model.predict(last_60_days)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Create a DataFrame for the predicted prices
    future_dates = pd.date_range(start=data.index[-1], periods=31, inclusive='right')
    predicted_df = pd.DataFrame(predicted_prices[0], index=future_dates, columns=['Predicted Close'])

    # Create an interactive plot for actual prices
    fig_actual = px.line(data, x=data.index, y='Close', title=f'Actual {stock_symbol} Stock Price')
    fig_actual.update_xaxes(title_text='Date')
    fig_actual.update_yaxes(title_text='Close Price')

    # Create an interactive plot for predicted prices
    fig_predicted = px.line(predicted_df, x=predicted_df.index, y='Predicted Close', title=f'Predicted {stock_symbol} Stock Price')
    fig_predicted.update_xaxes(title_text='Date')
    fig_predicted.update_yaxes(title_text='Predicted Close Price')

    # Convert the plots to HTML
    plot_actual_html = fig_actual.to_html(full_html=False)
    plot_predicted_html = fig_predicted.to_html(full_html=False)

    # Convert the data to an HTML table
    data_html = data.to_html(classes='table table-striped', border=0)

    # Render the plots and data in an HTML template
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Stock Price</title>
            <style>
                .scrollable-table {
                    height: 400px;
                    overflow-y: scroll;
                    border: 1px solid #ddd;
                }
                .table {
                    width: 100%;
                    border-collapse: collapse;
                }
                .table th, .table td {
                    padding: 8px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
            </style>
        </head>
        <body>
            <h1>Stock Price</h1>
            <form method="post">
                <label for="stock_symbol">Enter Stock Symbol:</label>
                <input type="text" id="stock_symbol" name="stock_symbol" value="{{ stock_symbol }}">
                <input type="submit" value="Submit">
            </form>
            <h1>Actual {{ stock_symbol }} Stock Price</h1>
            {{ plot_actual_html|safe }}
            <h1>Predicted {{ stock_symbol }} Stock Price</h1>
            {{ plot_predicted_html|safe }}
            <h1>Fetched Data</h1>
            <div class="scrollable-table">
                {{ data_html|safe }}
            </div>
        </body>
        </html>
    ''', stock_symbol=stock_symbol, plot_actual_html=plot_actual_html, plot_predicted_html=plot_predicted_html, data_html=data_html)

if __name__ == '__main__':
    app.run()