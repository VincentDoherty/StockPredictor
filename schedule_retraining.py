import schedule
import time
from retrain_model import retrain_model

def job():
    stock_symbol = 'GOOGL'  # Replace with the desired stock symbol
    retrain_model(stock_symbol)

# Schedule the job every week
schedule.every().week.do(job)

while True:
    schedule.run_pending()
    time.sleep(1)