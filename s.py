import yfinance as yf
import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image
from datetime import datetime

def download_data(symbol, start_date, end_date):
    """
    Downloads historical stock data from Yahoo Finance.

    Args:
        symbol: The stock symbol (e.g., AAPL)
        start_date: The starting date for data download (YYYY-MM-DD)
        end_date: The ending date for data download (YYYY-MM-DD)

    Returns:
        A Pandas DataFrame containing the downloaded data
    """
    data = yf.download(symbol, start=start_date, end=end_date)
    # Ensure specific columns are included for plotting
    data = data[["Open", "Close", "Volume", "High", "Low"]]
    return data

def calculate_macd(data):
    """
    Calculates the Moving Average Convergence Divergence (MACD) indicator.

    Args:
        data: A Pandas DataFrame containing the stock data

    Returns:
        A Pandas DataFrame with the MACD indicator added
    """
    data["MACD"] = data["Close"].ewm(span=12, adjust=False).mean() - data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Diff"] = data["MACD"] - data["MACD_Signal"]
    return data

def calculate_rsi(data):
    """
    Calculates the Relative Strength Index (RSI) indicator.

    Args:
        data: A Pandas DataFrame containing the stock data

    Returns:
        A Pandas DataFrame with the RSI indicator added
    """
    delta = data["Close"].diff()
    up, down = delta.clip(lower=0), delta.clip(upper=0).abs()
    ema_up, ema_down = up.ewm(span=14, adjust=False).mean(), down.ewm(span=14, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    data["RSI"] = rsi
    return data

def calculate_sma(data, window):
    return data["Close"].rolling(window=window).mean()

def calculate_ema(data, span):
    return data["Close"].ewm(span=span, adjust=False).mean()

def calculate_mfi(data, period):
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    raw_money_flow = typical_price * data["Volume"]

    # Calculate money flow
    money_flow = np.where(typical_price.diff() > 0, raw_money_flow.diff(), 0)

    # Calculate positive and negative money flow
    positive_flow = np.sum(np.where(money_flow > 0, money_flow, 0), axis=0)
    negative_flow = np.abs(np.sum(np.where(money_flow < 0, money_flow, 0), axis=0))

    # Calculate money flow ratio
    money_flow_ratio = np.divide(positive_flow, negative_flow + 1e-10)

    # Calculate MFI
    money_flow_index = 100 - (100 / (1 + money_flow_ratio))

    # Convert to Pandas Series before applying rolling
    data["MFI"] = pd.Series(money_flow_index).rolling(window=period).mean()

    return data

def calculate_bollinger_bands(data, window):
    sma = calculate_sma(data, window)
    rolling_std = data["Close"].rolling(window=window).std()
    data["Upper Band"] = sma + (2 * rolling_std)
    data["Lower Band"] = sma - (2 * rolling_std)
    return data

def calculate_stochastic_oscillator(data, k_period, d_period):
    lowest_low = data["Low"].rolling(window=k_period).min()
    highest_high = data["High"].rolling(window=k_period).max()
    data["%K"] = ((data["Close"] - lowest_low) / (highest_high - lowest_low)) * 100
    data["%D"] = data["%K"].rolling(window=d_period).mean()
    return data

def calculate_adx(data, window):
    high_low = data["High"] - data["Low"]
    high_close = np.abs(data["High"] - data["Close"].shift())
    low_close = np.abs(data["Low"] - data["Close"].shift())

    tr = np.maximum(np.maximum(high_low, high_close), low_close)
    data["+DM"] = np.where((data["High"] - data["High"].shift()) > (data["Low"].shift() - data["Low"]),
                           np.maximum(data["High"] - data["High"].shift(), 0), 0)
    data["-DM"] = np.where((data["Low"].shift() - data["Low"]) > (data["High"] - data["High"].shift()),
                           np.maximum(data["Low"].shift() - data["Low"], 0), 0)

    atr = tr.ewm(span=window, adjust=False).mean()
    plus_di = (data["+DM"].ewm(span=window, adjust=False).mean() / atr) * 100
    minus_di = (data["-DM"].ewm(span=window, adjust=False).mean() / atr) * 100

    data["ADX"] = np.abs((plus_di - minus_di) / (plus_di + minus_di)) * 100
    return data

def analyze_price_trends(data):
    """
    Analyzes the stock price trends and identifies potential buy and sell signals.

    Args:
        data: A Pandas DataFrame containing the calculated indicators

    Returns:
        A Pandas DataFrame with the Buy Signal and Sell Signal columns added
    """
    data["SMA_20"] = data["Close"].rolling(window=20).mean()
    data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

    # Make sure required columns are available
    required_columns = ["High", "Low", "Close"]
    if all(col in data.columns for col in required_columns):
        data = calculate_mfi(data, 14)

    data = calculate_bollinger_bands(data, 20)
    data = calculate_stochastic_oscillator(data, 14, 3)
    data = calculate_adx(data, 14)
    data["Buy Signal"] = np.where((data["Close"] > data["SMA_20"]) & (data["Close"] > data["EMA_20"]) & (data["RSI"] < 50),
                                  True, False)
    data["Sell Signal"] = np.where((data["Close"] < data["SMA_20"]) & (data["Close"] < data["EMA_20"]) & (data["RSI"] > 70),
                                   True, False)
    return data

def generate_strategy(data):
    """
    Generates a basic trading strategy based on the analyzed signals.

    Args:
        data: A Pandas DataFrame containing the Buy Signal and Sell Signal columns

    Returns:
        A Pandas DataFrame with the Strategy column added
    """
    data["Strategy"] = np.where(data["Buy Signal"], "Buy", np.where(data["Sell Signal"], "Sell", "Hold"))
    return data

def generate_graph(data, symbol):
    """
    Generates a chart displaying price, indicators, and buy/sell signals.

    Args:
        data: A Pandas DataFrame containing the processed data
        symbol: The stock symbol (e.g., AAPL)
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # Plot price and volume
    ax1.plot(data.index, data["Close"], color="blue", label="Close")
    ax1.bar(data.index, data["Volume"], color="grey", alpha=0.2, label="Volume")

    # Plot SMA and EMA
    ax1.plot(data.index, data["SMA_20"], color="green", label="SMA_20")
    ax1.plot(data.index, data["EMA_20"], color="red", label="EMA_20")

    # Plot Bollinger Bands
    ax1.plot(data.index, data["Upper Band"], color="orange", linestyle="--", label="Upper Band")
    ax1.plot(data.index, data["Lower Band"], color="orange", linestyle="--", label="Lower Band")

    # Plot MACD
    ax2.plot(data.index, data["MACD"], color="purple", label="MACD")
    ax2.plot(data.index, data["MACD_Signal"], color="brown", label="MACD Signal")

    # Plot RSI
    ax2.plot(data.index, data["RSI"], color="cyan", label="RSI")

    # Plot Stochastic Oscillator
    ax2.plot(data.index, data["%K"], color="pink", label="%K")
    ax2.plot(data.index, data["%D"], color="yellow", label="%D")

    # Plot ADX
    ax2.plot(data.index, data["ADX"], color="gray", label="ADX")

    # Add labels and title
    ax1.set_ylabel("Price and Volume")
    ax2.set_ylabel("Indicators")
    plt.title(f"{symbol} Stock Price with Indicators")

    # Add buy and sell signals
    buy_signals = data[data["Buy Signal"]].index
    sell_signals = data[data["Sell Signal"]].index
    plt.scatter(buy_signals, data.loc[buy_signals, "Close"], marker="^", color="green", label="Buy Signal")
    plt.scatter(sell_signals, data.loc[sell_signals, "Close"], marker="v", color="red", label="Sell Signal")

    # Add grid and legend
    plt.grid(True)
    plt.legend()


def save_to_excel(data, filename, symbol):
    """
    Saves the processed data and generated graph to an Excel file.

    Args:
        data: A Pandas DataFrame containing the processed data
        filename: The name of the Excel file to save
        symbol: The stock symbol (e.g., AAPL)
    """
    wb = openpyxl.Workbook()

    # Create sheets for data and graph
    sheet_data = wb.active
    sheet_data.title = "Data"
    sheet_graph = wb.create_sheet(title="Graph")

    # Write data to sheet
    for row in dataframe_to_rows(data, index=True, header=True):
        sheet_data.append(row)

    # Save graph as PNG and insert into sheet
    plt.savefig("temp.png")
    img = Image("temp.png")
    img.anchor = "A10"
    sheet_graph.add_image(img)

    # Save workbook
    wb.save(filename)

    # Show the plot after saving the Excel file
    plt.show()

def main():
    symbol = input("Enter stock symbol: ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = datetime.today().strftime("%Y-%m-%d")
    filename = f"{symbol}_{start_date}_{end_date}.xlsx"

    data = download_data(symbol, start_date, end_date)
    data = calculate_macd(data.copy())
    data = calculate_rsi(data.copy())
    data = analyze_price_trends(data.copy())
    data = generate_strategy(data.copy())
    generate_graph(data.copy(), symbol)
    save_to_excel(data.copy(), filename, symbol)

    print(f"Data and graph saved to: {filename}")

if __name__ == "__main__":
    main()
