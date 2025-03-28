import psycopg2

def create_tables():
    commands = [
        """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            risk_tolerance VARCHAR(50),
            investment_preferences TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS portfolios (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            name VARCHAR(255) NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS transactions (
            id SERIAL PRIMARY KEY,
            portfolio_id INTEGER REFERENCES portfolios(id),
            stock_symbol VARCHAR(10) NOT NULL,
            transaction_type VARCHAR(4) CHECK (transaction_type IN ('buy', 'sell')),
            quantity INTEGER NOT NULL,
            price NUMERIC NOT NULL,
            date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS stock_data (
            stock_symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            close NUMERIC NOT NULL,
            PRIMARY KEY (stock_symbol, date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS stock_predictions (
            stock_symbol VARCHAR(10) NOT NULL,
            date DATE NOT NULL,
            predicted_close NUMERIC NOT NULL,
            PRIMARY KEY (stock_symbol, date)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS notifications (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            message TEXT NOT NULL,
            date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            user_id INTEGER REFERENCES users(id),
            feedback TEXT NOT NULL,
            date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS stock_models (
            stock_symbol VARCHAR(10) PRIMARY KEY,
            model BYTEA NOT NULL,
            last_updated TIMESTAMP NOT NULL
        )
        """
    ]

    conn = None
    try:
        conn = psycopg2.connect(
            dbname='investmentdb',
            user='postgres',
            password='vdonkeY800',
            host='localhost',
            port='5432'
        )
        cursor = conn.cursor()
        for command in commands:
            cursor.execute(command)
        cursor.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

if __name__ == '__main__':
    create_tables()