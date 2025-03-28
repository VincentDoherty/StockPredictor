import numpy as np
from db_utils import get_db_connection, get_stock_data

def calculate_volatility(portfolio_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT stock_symbol, quantity
        FROM transactions
        WHERE portfolio_id = %s
    """, (portfolio_id,))
    transactions = cursor.fetchall()
    cursor.close()
    conn.close()

    stock_volatilities = []
    for transaction in transactions:
        stock_symbol, quantity = transaction
        stock_data = get_stock_data(stock_symbol)
        if not stock_data.empty:
            returns = stock_data['close'].pct_change().dropna()
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            stock_volatilities.append(volatility * quantity)

    portfolio_volatility = np.sum(stock_volatilities)
    return portfolio_volatility

def assess_risk(portfolio_id):
    volatility = calculate_volatility(portfolio_id)
    if volatility < 0.1:
        return 'Low Risk'
    elif volatility < 0.2:
        return 'Medium Risk'
    else:
        return 'High Risk'