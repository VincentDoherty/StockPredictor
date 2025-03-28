import psycopg2
import pandas as pd

def get_db_connection():
    conn = psycopg2.connect(
        dbname='investmentdb',
        user='postgres',
        password='vdonkeY800',
        host='localhost',
        port='5432'
    )
    return conn

def get_stock_data(stock_symbol):
    conn = get_db_connection()
    query = f"SELECT date, close FROM stock_data WHERE stock_symbol = '{stock_symbol}' ORDER BY date"
    stock_data = pd.read_sql(query, conn)
    conn.close()
    return stock_data