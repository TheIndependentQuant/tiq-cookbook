```python
# 10 - Keltner Band Trend Break
# Source: The Independent Quant | theindependentquant.com
# The Keltner Band Trend Break strategy is a trend-following strategy that uses Keltner Channels to identify potential trend reversals in the SPDR S&P 500 ETF (SPY). It calculates these channels using a 20-day EMA and a 2.0 multiplier of the Average True Range (ATR) over the same period. The strategy generates a buy signal when the price breaks above the upper band and a sell signal when it breaks below the lower band, indicating potential upward or downward trends, respectively.
# References:
# (No external references)
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and plot flag. For example: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import pandas as pd
import numpy as np
import yfinance as yf
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
EMA_PERIOD = 20
ATR_MULTIPLIER = 2.0

def download_data(ticker, start, end):
    """Download historical data for a given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute Keltner Band Trend Break signals."""
    df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    df['ATR'] = df['High'].rolling(window=EMA_PERIOD).max() - df['Low'].rolling(window=EMA_PERIOD).min()
    df['Upper_Band'] = df['EMA'] + (df['ATR'] * ATR_MULTIPLIER)
    df['Lower_Band'] = df['EMA'] - (df['ATR'] * ATR_MULTIPLIER)
    
    df['Signal'] = 0
    df.loc[df['Close'] > df['Upper_Band'].shift(1), 'Signal'] = 1
    df.loc[df['Close'] < df['Lower_Band'].shift(1), 'Signal'] = -1
    return df

def backtest(df):
    """Perform backtest on the signals."""
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1]) ** (252 / len(df)) - 1
    sharpe = df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * np.sqrt(252)
    downside_std = df[df['Strategy_Returns'] < 0]['Strategy_Returns'].std() * np.sqrt(252)
    sortino = df['Strategy_Returns'].mean() / downside_std
    max_drawdown = ((df['Cumulative'].cummax() - df['Cumulative']).max()) / df['Cumulative'].cummax().max()
    calmar = cagr / max_drawdown
    
    return {
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print formatted performance table."""
    print(f"Performance for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot equity curve vs buy-and-hold."""
    df['Buy_and_Hold'] = (1 + df['Close'].pct_change()).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot(df['Buy_and_Hold'], label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print current signal and position."""
    df = download_data(ticker, datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    current_signal = df['Signal'].iloc[-1]
    position = "Long" if current_signal == 1 else "Short" if current_signal == -1 else "Neutral"
    print(f"Today's signal for {ticker}: {position}")

def main():
    parser = argparse.ArgumentParser(description='Keltner Band Trend Break Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', default='2010-01-01', help='Start date (default: 2010-01-01)')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'), help='End date (default: today)')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve if set')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, "Keltner Band Trend Break")

    todays_signal(args.ticker)

if __name__ == "__main__":
    main()
```