import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import psycopg2
import pickle

# Database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            dbname='investmentdb',
            user='postgres',
            password='vdonkeY800',
            host='localhost',
            port='5432'
        )
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

# Retrieve stock data from the database
def get_stock_data(stock_symbol):
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()
    cursor = conn.cursor()
    cursor.execute("SELECT date, close FROM stock_data WHERE stock_symbol = %s ORDER BY date", (stock_symbol,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows, columns=['Date', 'Close']).set_index('Date')

# Save the trained model to the database
def save_model_to_db(stock_symbol, model):
    conn = get_db_connection()
    if conn is None:
        return
    cursor = conn.cursor()
    try:
        model_binary = pickle.dumps(model)
        cursor.execute(
            "INSERT INTO stock_models (stock_symbol, model, last_updated) VALUES (%s, %s, %s) ON CONFLICT (stock_symbol) DO UPDATE SET model = EXCLUDED.model, last_updated = EXCLUDED.last_updated",
            (stock_symbol, model_binary, datetime.now())
        )
        conn.commit()
        print(f"Model for {stock_symbol} saved to the database.")
    except Exception as e:
        print(f"Error saving model to the database: {e}")
    finally:
        cursor.close()
        conn.close()

# Load the trained model from the database
def load_model_from_db(stock_symbol):
    conn = get_db_connection()
    if conn is None:
        return None, None
    cursor = conn.cursor()
    cursor.execute("SELECT model, last_updated FROM stock_models WHERE stock_symbol = %s", (stock_symbol,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    if result:
        model = pickle.loads(result[0])
        last_updated = result[1]
        return model, last_updated
    return None, None

# Retrain the model
def retrain_model(stock_symbol):
    # Fetch the most up-to-date stock data
    existing_data = get_stock_data(stock_symbol)
    last_date = existing_data.index[-1] if not existing_data.empty else '2020-01-01'
    new_data = yf.download(stock_symbol, start=last_date, end=datetime.now().strftime('%Y-%m-%d'))
    if not new_data.empty:
        new_data = new_data[['Close']]
        existing_data = pd.concat([existing_data, new_data])

    # Prepare the data for the LSTM model
    data = existing_data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create the training data using all available data
    train_data = []
    target_data = []
    for i in range(365, len(scaled_data) - 30):
        train_data.append(scaled_data[i - 365:i, 0])  # Append last 365 elements
        target_data.append(scaled_data[i:i + 30, 0])  # Append next 30 elements
    train_data, target_data = np.array(train_data), np.array(target_data)

    # Reshape the data for the LSTM model
    train_data = np.reshape(train_data, (train_data.shape[0], train_data.shape[1], 1))

    # Load existing model if available
    model, last_updated = load_model_from_db(stock_symbol)
    if model is None or (datetime.now() - last_updated).days > 7:
        # Build the optimized LSTM model if no existing model or model is outdated
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(train_data.shape[1], 1)))
        model.add(Dropout(0.3))  # Adjusted dropout rate
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.3))  # Adjusted dropout rate
        model.add(LSTM(units=100))
        model.add(Dropout(0.3))  # Adjusted dropout rate
        model.add(Dense(30))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model with more epochs and adjusted batch size
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
        model.fit(train_data, target_data, epochs=10, batch_size=32, callbacks=[early_stopping])

        # Save the trained model to the database
        save_model_to_db(stock_symbol, model)

# Example usage
if __name__ == '__main__':
    stock_symbol = 'GOOGL'  # Replace with the desired stock symbol
    retrain_model(stock_symbol)