```python
# 01 - SMA Crossover (10-50)
# Source: The Independent Quant | theindependentquant.com
# Description: The SMA Crossover (10-50) strategy is a trend-following method that uses two simple moving averages
# to identify buy and sell signals for SPY. A buy signal is generated when the 10-day SMA crosses above the 50-day SMA,
# indicating a potential upward trend. Conversely, a sell signal is generated when the 10-day SMA crosses below the 50-day SMA,
# indicating a potential downward trend. This strategy aims to capture medium-term trends by smoothing out short-term fluctuations.
# References:
# https://www.sma-america.com/
# Usage: Run the script with optional arguments --ticker, --start, --end, and --plot to customize the analysis.

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Constants
SHORT_WINDOW = 10
LONG_WINDOW = 50

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute buy/sell signals based on SMA crossover."""
    df['SMA10'] = df['Close'].rolling(window=SHORT_WINDOW, min_periods=1).mean()
    df['SMA50'] = df['Close'].rolling(window=LONG_WINDOW, min_periods=1).mean()
    df['Signal'] = 0
    df['Signal'][SHORT_WINDOW:] = np.where(df['SMA10'][SHORT_WINDOW:] > df['SMA50'][SHORT_WINDOW:], 1, 0)
    df['Position'] = df['Signal'].shift(1)
    return df

def backtest(df):
    """Backtest the strategy and calculate returns."""
    df['Market_Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Market_Returns'] * df['Position']
    df['Cumulative_Market'] = (1 + df['Market_Returns']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative_Strategy'].iloc[-1])**(252/len(df)) - 1
    sharpe_ratio = np.sqrt(252) * (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std())
    downside_std = df['Strategy_Returns'][df['Strategy_Returns'] < 0].std()
    sortino_ratio = np.sqrt(252) * (df['Strategy_Returns'].mean() / downside_std)
    max_drawdown = (df['Cumulative_Strategy'] / df['Cumulative_Strategy'].cummax() - 1).min()
    calmar_ratio = cagr / abs(max_drawdown)
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print performance results."""
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve of the strategy vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative_Market'], label='Buy and Hold', color='blue')
    plt.plot(df['Cumulative_Strategy'], label=strategy_name, color='red')
    plt.title(f"{strategy_name} vs Buy and Hold for {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Print today's signal based on recent data."""
    df = download_data(ticker, start=datetime.now().strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    signal = df['Signal'].iloc[-1]
    position = df['Position'].iloc[-1]
    print(f"Today's Signal for {ticker}: {'Buy' if signal == 1 else 'Sell'}")
    print(f"Current Position: {'Long' if position == 1 else 'Out'}")

def main():
    parser = argparse.ArgumentParser(description='SMA Crossover (10-50) Strategy')
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
        plot_results(df, args.ticker, "SMA Crossover (10-50)")

    todays_signal(args.ticker)

if __name__ == '__main__':
    main()
```