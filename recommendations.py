from app import get_db_connection


def get_investment_recommendations(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT risk_tolerance FROM users WHERE id = %s", (user_id,))
    risk_tolerance = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    if risk_tolerance == 'Low':
        recommended_stocks = ['AAPL', 'MSFT', 'GOOGL']
    elif risk_tolerance == 'Medium':
        recommended_stocks = ['TSLA', 'AMZN', 'FB']
    else:
        recommended_stocks = ['BTC-USD', 'ETH-USD', 'DOGE-USD']

    return recommended_stocks