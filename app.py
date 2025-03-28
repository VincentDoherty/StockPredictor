import os
import yfinance as yf
import plotly.express as px
from flask import Flask, render_template_string, request, redirect, url_for, flash, Response, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import psycopg2
from datetime import datetime
from werkzeug.security import generate_password_hash
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from db_utils import get_db_connection, get_stock_data
from retrain_model import retrain_model, load_model_from_db
from user import User
from backtesting import backtest_strategy, example_strategy
from risk_assessment import assess_risk
from recommendations import get_investment_recommendations
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5174"}}, supports_credentials=True)
app.secret_key = os.urandom(24)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error=error), 500

@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error=error), 404

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, password_hash FROM users WHERE id = %s", (user_id,))
    user_data = cursor.fetchone()
    cursor.close()
    conn.close()
    if user_data:
        return User(user_data[0], user_data[1], user_data[2])
    return None

# Store stock data in the database
def store_stock_data(stock_symbol, data):
    conn = get_db_connection()
    cursor = conn.cursor()
    for date, row in data.iterrows():
        cursor.execute(
            """
            INSERT INTO stock_data (stock_symbol, date, close)
            VALUES (%s, %s, %s)
            ON CONFLICT (stock_symbol, date)
            DO UPDATE SET close = EXCLUDED.close
            """,
            (stock_symbol, date, row['Close'].item())
        )
    conn.commit()
    cursor.close()
    conn.close()

# Retrieve stock data from the database
def get_stock_data(stock_symbol):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT date, close FROM stock_data WHERE stock_symbol = %s ORDER BY date", (stock_symbol,))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows, columns=['Date', 'Close']).set_index('Date')

# Store predictions in the database
def store_predictions(stock_symbol, predictions):
    conn = get_db_connection()
    cursor = conn.cursor()
    for date, price in predictions.iterrows():
        cursor.execute(
            "INSERT INTO stock_predictions (stock_symbol, date, predicted_close) VALUES (%s, %s, %s) ON CONFLICT (stock_symbol, date) DO UPDATE SET predicted_close = EXCLUDED.predicted_close",
            (stock_symbol, date, float(price['Predicted Close']))
        )
    conn.commit()
    cursor.close()
    conn.close()

# Portfolio management
def create_portfolio_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            name VARCHAR(255) NOT NULL
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def create_transactions_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            portfolio_id INTEGER REFERENCES portfolios(id),
            stock_symbol VARCHAR(10) NOT NULL,
            transaction_type VARCHAR(4) CHECK (transaction_type IN ('buy', 'sell')),
            quantity INTEGER NOT NULL,
            price NUMERIC NOT NULL,
            date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def store_transaction(portfolio_id, stock_symbol, transaction_type, quantity, price):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO transactions (portfolio_id, stock_symbol, transaction_type, quantity, price)
        VALUES (%s, %s, %s, %s, %s)
    """, (portfolio_id, stock_symbol, transaction_type, quantity, price))
    conn.commit()
    cursor.close()
    conn.close()

def calculate_profit_loss(portfolio_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT transaction_type, quantity, price
        FROM transactions
        WHERE portfolio_id = %s
    """, (portfolio_id,))
    transactions = cursor.fetchall()
    cursor.close()
    conn.close()

    profit_loss = 0
    for transaction in transactions:
        if transaction[0] == 'buy':
            profit_loss -= transaction[1] * transaction[2]
        elif transaction[0] == 'sell':
            profit_loss += transaction[1] * transaction[2]
    return profit_loss

def get_portfolios(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM portfolios WHERE user_id = %s", (user_id,))
    portfolios = cursor.fetchall()
    cursor.close()
    conn.close()
    return portfolios

def get_transactions(portfolio_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT stock_symbol, transaction_type, quantity, price, date FROM transactions WHERE portfolio_id = %s", (portfolio_id,))
    transactions = cursor.fetchall()
    cursor.close()
    conn.close()
    return transactions

# Login routes
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    password_hash = generate_password_hash(password)
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            (username, password_hash)
        )
        conn.commit()
        return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        cursor.close()
        conn.close()

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        username = data['username']
        password = data['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash FROM users WHERE username = %s", (username,))
        user_data = cursor.fetchone()
        cursor.close()
        conn.close()

        if user_data:
            user = User(user_data[0], user_data[1], user_data[2])
            if user.check_password(password):
                login_user(user)
                return jsonify({'message': 'Login successful'}), 200
        return jsonify({'message': 'Invalid username or password'}), 401
    except Exception as e:
        logging.error(f"Error during login: {e}")
        return jsonify({'error': 'An error occurred during login'}), 500

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200

@app.route('/api/status', methods=['GET'])
def status():
    try:
        if current_user.is_authenticated:
            return jsonify({'status': 'logged_in', 'user': current_user.username}), 200
        else:
            return jsonify({'status': 'not_logged_in'}), 200
    except Exception as e:
        logging.error(f"Error in /status endpoint: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

# Dashboard routes
@app.route('/admin', methods=['GET'])
@login_required
def admin_dashboard():
    if not current_user.has_permission('admin'):
        return redirect(url_for('dashboard'))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, role FROM users")
    users = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(users)

@app.route('/dashboard')
@login_required
def dashboard():
    portfolios = get_portfolios(current_user.id)
    portfolio_performances = {portfolio[1]: calculate_profit_loss(portfolio[0]) for portfolio in portfolios}
    recent_transactions = []
    for portfolio in portfolios:
        transactions = get_transactions(portfolio[0])
        recent_transactions.extend(transactions)

    return jsonify({
        'portfolio_performances': portfolio_performances,
        'recent_transactions': recent_transactions
    })

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        risk_tolerance = request.form['risk_tolerance']
        investment_preferences = request.form['investment_preferences']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET risk_tolerance = %s, investment_preferences = %s WHERE id = %s",
                       (risk_tolerance, investment_preferences, current_user.id))
        conn.commit()
        cursor.close()
        conn.close()
        flash('Profile updated successfully')
    return render_template('profile.html', current_user=current_user)

@app.route('/api/stock', methods=['POST'])
def stock():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        stock_symbol = request.json.get('stock_symbol', 'GOOGL')
        retrain_model(stock_symbol)

        existing_data = get_stock_data(stock_symbol)
        last_date = existing_data.index[-1] if not existing_data.empty else '2020-01-01'
        new_data = yf.download(stock_symbol, start=last_date, end=datetime.now().strftime('%Y-%m-%d'))
        if not new_data.empty:
            new_data = new_data[['Close']]
            store_stock_data(stock_symbol, new_data)
            existing_data = get_stock_data(stock_symbol)

        data = existing_data[['Close']]

        # Handle missing values
        data = data.dropna()

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        model, _ = load_model_from_db(stock_symbol)

        last_60_days = scaled_data[-60:]
        last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
        predicted_prices = model.predict(last_60_days)
        predicted_prices = scaler.inverse_transform(predicted_prices)

        future_dates = pd.date_range(start=data.index[-1], periods=31, inclusive='right')
        predicted_df = pd.DataFrame(predicted_prices[0], index=future_dates, columns=['Predicted Close'])
        store_predictions(stock_symbol, predicted_df)

        actual_prices = data['Close'].tolist()
        actual_dates = data.index.to_list()
        actual_dates = [date.strftime('%Y-%m-%d') for date in actual_dates]
        predicted_prices = predicted_df['Predicted Close'].tolist()
        predicted_dates = predicted_df.index.to_list()
        predicted_dates = [date.strftime('%Y-%m-%d') for date in predicted_dates]

        return jsonify({
            'stock_symbol': stock_symbol,
            'actual_prices': actual_prices,
            'actual_dates': actual_dates,
            'predicted_prices': predicted_prices,
            'predicted_dates': predicted_dates
        })
    except Exception as e:
        logging.error(f"Error in /stock endpoint: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500

# Portfolio routes
@app.route('/create_portfolio', methods=['POST'])
@login_required
def create_portfolio():
        portfolio_name = request.form['portfolio_name']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO portfolios (user_id, name) VALUES (%s, %s)", (current_user.id, portfolio_name))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'message': 'Portfolio created successfully'})

@app.route('/view_portfolios')
@login_required
def view_portfolios():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM portfolios WHERE user_id = %s", (current_user.id,))
    portfolios = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(portfolios)

@app.route('/view_portfolio/<int:portfolio_id>')
@login_required
def view_portfolio(portfolio_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT stock_symbol, transaction_type, quantity, price, date FROM transactions WHERE portfolio_id = %s", (portfolio_id,))
    transactions = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('view_portfolio.html', transactions=transactions)

@app.route('/portfolio_performance/<int:portfolio_id>')
@login_required
def portfolio_performance(portfolio_id):
    profit_loss = calculate_profit_loss(portfolio_id)
    return render_template_string('''
        <h1>Portfolio Performance</h1>
        <p>Profit/Loss: {{ profit_loss }}</p>
    ''', profit_loss=profit_loss)

@app.route('/export_portfolio/<int:portfolio_id>', methods=['GET'])
@login_required
def export_portfolio(portfolio_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT stock_symbol, transaction_type, quantity, price, date FROM transactions WHERE portfolio_id = %s", (portfolio_id,))
    transactions = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(transactions, columns=['Stock Symbol', 'Transaction Type', 'Quantity', 'Price', 'Date'])
    csv_data = df.to_csv(index=False)
    return Response(csv_data, mimetype='text/csv', headers={'Content-Disposition': 'attachment;filename=portfolio.csv'})

# Notification routes
def send_notification(user_id, message):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO notifications (user_id, message) VALUES (%s, %s)", (user_id, message))
    conn.commit()
    cursor.close()
    conn.close()

@app.route('/notifications')
@login_required
def notifications():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT message FROM notifications WHERE user_id = %s", (current_user.id,))
    notifications = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(notifications)

@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    if request.method == 'POST':
        feedback_text = request.form['feedback']
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO feedback (user_id, feedback) VALUES (%s, %s)", (current_user.id, feedback_text))
        conn.commit()
        cursor.close()
        conn.close()
        flash('Feedback submitted successfully')
    return render_template('feedback.html')

# Backtesting routes
from flask import Flask, request, jsonify
import pandas as pd
import logging

app = Flask(__name__)


@app.route('/api/backtest', methods=['POST'])
def backtest():
    try:
        data = request.get_json()
        stock_symbol = data['stock_symbol']
        start_date = data['start_date']
        end_date = data['end_date']

        # Assuming you have a function to get the stock data
        stock_data = get_stock_data(stock_symbol, start_date, end_date)

        # Ensure the 'Date' column is set as the index
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)

        # Perform your backtest logic here
        cumulative_returns = calculate_cumulative_returns(stock_data)

        return jsonify({'cumulative_returns': cumulative_returns})
    except Exception as e:
        logging.error(f"Error in /api/backtest endpoint: {e}")
        return jsonify({'error': 'Internal Server Error'}), 500



def calculate_cumulative_returns(stock_data):
    # Dummy function to simulate calculating cumulative returns
    # Replace this with actual calculation logic
    stock_data['Cumulative Returns'] = stock_data['Close'].pct_change().cumsum()
    return stock_data['Cumulative Returns'].to_dict()


# Risk assessment routes
@app.route('/portfolio/<int:portfolio_id>/risk', methods=['GET'])
@login_required
def portfolio_risk(portfolio_id):
    risk_level = assess_risk(portfolio_id)
    return jsonify({
        'portfolio_id': portfolio_id,
        'risk_level': risk_level
    })

# Investment recommendation routes
@app.route('/recommendations', methods=['GET'])
@login_required
def recommendations():
    recommended_stocks = get_investment_recommendations(current_user.id)
    return jsonify({
        'recommended_stocks': recommended_stocks
    })

# Error handling and logging
@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Error: {e}")
    return render_template('error.html'), 500

# Stock news route
import requests

def get_stock_news(stock_symbol):
    api_key = 'your_api_key'
    url = f'https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={api_key}'
    response = requests.get(url)
    return response.json().get('articles', [])

@app.route('/stock_news/<stock_symbol>')
@login_required
def stock_news(stock_symbol):
    news_articles = get_stock_news(stock_symbol)
    return render_template('stock_news.html', stock_symbol=stock_symbol, news_articles=news_articles)

if __name__ == '__main__':
    app.run()