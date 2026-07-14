```python
# 09 - Trend Line Break with Confirmation
# Source: The Independent Quant | theindependentquant.com
# This strategy identifies potential entry points by detecting when the price of SPY breaks through a predefined trend line, with confirmation from the Relative Strength Index (RSI). A break above a descending trend line or below an ascending trend line signals a potential reversal or continuation of the trend. The confirmation step requires the RSI to cross above or below a specific threshold to filter out false signals, enhancing the reliability of trades.
# References:
# (No external references)
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and plot flag. Example: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-10 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
RSI_PERIOD = 14
RSI_THRESHOLD = 50

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the trend line break with confirmation signal."""
    df['RSI'] = compute_rsi(df['Close'], RSI_PERIOD)
    df['Signal'] = np.where((df['Close'] > df['Close'].shift(1)) & (df['RSI'] > RSI_THRESHOLD), 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def compute_rsi(series, period):
    """Calculate the Relative Strength Index (RSI)."""
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest(df):
    """Backtest the strategy and calculate returns."""
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Signal'] * df['Market_Return']
    df['Cumulative_Market'] = (1 + df['Market_Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative_Strategy'].iloc[-1])**(252/len(df)) - 1
    sharpe = (df['Strategy_Return'].mean() / df['Strategy_Return'].std()) * np.sqrt(252)
    downside_std = df['Strategy_Return'][df['Strategy_Return'] < 0].std()
    sortino = (df['Strategy_Return'].mean() / downside_std) * np.sqrt(252)
    max_drawdown = ((df['Cumulative_Strategy'].cummax() - df['Cumulative_Strategy']).max())
    calmar = cagr / max_drawdown
    return {
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print the performance results."""
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve of the strategy vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative_Market'], label='Buy & Hold', linestyle='--')
    plt.plot(df['Cumulative_Strategy'], label=strategy_name)
    plt.title(f'{strategy_name} vs Buy & Hold on {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print the current signal and position."""
    df = download_data(ticker, start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    current_signal = df['Signal'].iloc[-1]
    print(f"Today's signal for {ticker} is: {'Buy' if current_signal == 1 else 'Hold/Sell'}")

def main():
    """Main function to wire everything together."""
    parser = argparse.ArgumentParser(description='Trend Line Break with Confirmation Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol to analyze')
    parser.add_argument('--start', default='2010-01-01', help='Start date for historical data')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help='End date for historical data')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve if set')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, 'Trend Line Break with Confirmation')

if __name__ == "__main__":
    main()
```