```python
# 45 - Volume Dry-Up Filter for Breakout Entry
# Source: The Independent Quant | theindependentquant.com
# This strategy identifies potential breakout opportunities in SPY by analyzing volume patterns. It detects periods of significantly reduced trading volume, known as "volume dry-up," which often precede a breakout. By entering a position when these conditions are met, the strategy aims to capture subsequent price movements, leveraging market inefficiencies and behavioral finance principles.

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
VOLUME_LOOKBACK = 20
VOLUME_THRESHOLD = 0.5

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the Volume Dry-Up signal."""
    df['Avg_Volume'] = df['Volume'].rolling(window=VOLUME_LOOKBACK).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Avg_Volume']
    df['Signal'] = np.where(df['Volume_Ratio'] < VOLUME_THRESHOLD, 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy."""
    df['Strategy_Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    total_return = df['Cumulative'].iloc[-1] - 1
    cagr = (df['Cumulative'].iloc[-1])**(252/len(df)) - 1
    sharpe = np.sqrt(252) * df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()
    downside_std = df[df['Strategy_Returns'] < 0]['Strategy_Returns'].std()
    sortino = np.sqrt(252) * df['Strategy_Returns'].mean() / downside_std
    max_drawdown = (df['Cumulative'].cummax() - df['Cumulative']).max()
    calmar = cagr / max_drawdown
    return {'Total Return': total_return, 'CAGR': cagr, 'Sharpe': sharpe, 'Sortino': sortino, 'Calmar': calmar, 'Max Drawdown': max_drawdown}

def print_results(perf, ticker):
    """Print formatted performance table."""
    print(f"Performance for {ticker} Strategy:")
    for key, value in perf.items():
        print(f"{key}: {value:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot equity curve vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label='Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} - {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print current signal and position."""
    df = download_data(ticker, start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    current_signal = df['Signal'].iloc[-1]
    print(f"Today's signal for {ticker}: {'Buy' if current_signal == 1 else 'No Position'}")

def main():
    parser = argparse.ArgumentParser(description='Volume Dry-Up Filter for Breakout Entry Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, "Volume Dry-Up Filter for Breakout Entry")

if __name__ == "__main__":
    main()
```