import yfinance as yf
from datetime import datetime as dt
from datetime import timedelta as delta
import os
import pandas as pd

def get_stock_info(stock, start_date, end_date, save_to_disk=False):
    try:
        df = yf.download(f"{stock}.NS", start=start_date, end=end_date, progress=False)
        if save_to_disk:
            path = './csv'
            try:
                os.mkdir(path)
            except OSError as error:
                pass
            df.to_csv(f'{path}/{stock}.csv', index=True)  # Set index to True to include DateTimeIndex
        return df
    except Exception as e:
        print(f"Error fetching or saving data for {stock}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error

# Read stock symbols from CSV file
csv_file_path = 'E:\Mini Project\Stock\stock_symbols.csv'  # Replace with the actual path to your CSV file
df_symbols = pd.read_csv(csv_file_path)

# Ask for the start date only once
start_date = input("Enter a start date in YY-MM-DD format: ")
end_date = (dt.now() + delta(1)).strftime('%Y-%m-%d')

# Iterate through each stock symbol and fetch stock data
for symbol in df_symbols['Symbol']:
    get_stock_info(symbol, start_date, end_date, save_to_disk=True)
