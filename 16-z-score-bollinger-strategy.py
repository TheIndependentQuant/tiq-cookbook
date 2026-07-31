```python
# 16 - Z-Score Bollinger Strategy
# Source: The Independent Quant | theindependentquant.com
# The Z-Score Bollinger Strategy is a mean reversion trading strategy for the SPY ETF, which tracks the S&P 500 index. 
# It uses Bollinger Bands and the Z-Score to generate trading signals, predicting that significant deviations from the average 
# price will revert back to the mean. A high Z-Score suggests overbought conditions, while a low Z-Score indicates oversold 
# conditions, prompting trades that bet on price reversion.
# References:
# (No external references)
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and plot flag.

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Constants
LOOKBACK = 20
STD_DEV = 2
Z_SCORE_THRESHOLD = 1.5

def download_data(ticker, start, end):
    """Download historical data for a given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the trading signal using Z-Score Bollinger Strategy."""
    df['SMA'] = df['Close'].rolling(window=LOOKBACK).mean()
    df['STD'] = df['Close'].rolling(window=LOOKBACK).std()
    df['Upper Band'] = df['SMA'] + (STD_DEV * df['STD'])
    df['Lower Band'] = df['SMA'] - (STD_DEV * df['STD'])
    df['Z-Score'] = (df['Close'] - df['SMA']) / df['STD']
    
    df['Signal'] = 0
    df.loc[df['Z-Score'] > Z_SCORE_THRESHOLD, 'Signal'] = -1
    df.loc[df['Z-Score'] < -Z_SCORE_THRESHOLD, 'Signal'] = 1
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy and compute returns."""
    df['Strategy Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative Strategy'] = (1 + df['Strategy Returns']).cumprod()
    df['Cumulative Market'] = (1 + df['Close'].pct_change()).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative Strategy'].iloc[-1]) ** (252 / len(df)) - 1
    strategy_returns = df['Strategy Returns'].dropna()
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    downside_std = strategy_returns[strategy_returns < 0].std()
    sortino = strategy_returns.mean() / downside_std * np.sqrt(252)
    max_drawdown = (df['Cumulative Strategy'] / df['Cumulative Strategy'].cummax() - 1).min()
    calmar = cagr / abs(max_drawdown)
    
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print the performance metrics."""
    print(f"Performance for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve of the strategy vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative Strategy'], label='Strategy')
    plt.plot(df.index, df['Cumulative Market'], label='Buy and Hold')
    plt.title(f'{strategy_name} on {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print the current signal and position."""
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - pd.DateOffset(days=LOOKBACK * 2)).strftime('%Y-%m-%d')
    df = download_data(ticker, start, end)
    df = compute_signal(df)
    current_signal = df['Signal'].iloc[-1]
    print(f"Today's signal for {ticker}: {current_signal}")

def main():
    parser = argparse.ArgumentParser(description='Z-Score Bollinger Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', default='2010-01-01', help='Start date (default: 2010-01-01)')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'), help='End date (default: today)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve if set')
    
    args = parser.parse_args()
    
    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)
    
    if args.plot:
        plot_results(df, args.ticker, "Z-Score Bollinger Strategy")
    
    todays_signal(args.ticker)

if __name__ == "__main__":
    main()
```