```python
# 47 - Adaptive Stop Loss via ATR
# Source: The Independent Quant | theindependentquant.com
# The Adaptive Stop Loss via ATR strategy is designed to manage risk dynamically when trading the SPDR S&P 500 ETF Trust (SPY). This strategy utilizes the Average True Range (ATR), a volatility indicator, to set stop-loss levels that adapt to changing market conditions. Specifically, the ATR is calculated over a 14-day period to gauge market volatility. The stop-loss level is set at a multiple of the ATR value, allowing for wider stops during volatile periods and tighter stops when the market is calm. This approach aims to prevent premature exits during normal market fluctuations while protecting against significant losses.
# References:
# (No external references)
# Usage instructions: Run this script with optional arguments for ticker, start date, end date, and plot flag. Example: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-15 --plot

import pandas as pd
import numpy as np
import yfinance as yf
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5

def download_data(ticker, start, end):
    """Download historical data for a given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the trading signal based on Adaptive Stop Loss via ATR."""
    df['ATR'] = df['High'].rolling(window=ATR_PERIOD).max() - df['Low'].rolling(window=ATR_PERIOD).min()
    df['ATR'] = df['ATR'].rolling(window=ATR_PERIOD).mean()
    df['Stop Loss'] = df['Close'] - ATR_MULTIPLIER * df['ATR']
    df['Signal'] = np.where(df['Close'].shift(1) > df['Stop Loss'].shift(1), 1, 0)
    return df

def backtest(df):
    """Backtest the strategy and calculate strategy returns and cumulative returns."""
    df['Strategy Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
    df['Cumulative Returns'] = (1 + df['Strategy Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative Returns'].iloc[-1]) ** (252.0/len(df)) - 1
    sharpe = df['Strategy Returns'].mean() / df['Strategy Returns'].std() * np.sqrt(252)
    sortino = df['Strategy Returns'].mean() / df[df['Strategy Returns'] < 0]['Strategy Returns'].std() * np.sqrt(252)
    max_drawdown = ((df['Cumulative Returns'].cummax() - df['Cumulative Returns']) / df['Cumulative Returns'].cummax()).max()
    calmar = cagr / max_drawdown
    return {'CAGR': cagr, 'Sharpe': sharpe, 'Sortino': sortino, 'Calmar': calmar, 'Max Drawdown': max_drawdown}

def print_results(perf, ticker):
    """Print the performance results in a formatted table."""
    print(f"Performance Metrics for {ticker}:")
    print(f"CAGR: {perf['CAGR']:.2%}")
    print(f"Sharpe Ratio: {perf['Sharpe']:.2f}")
    print(f"Sortino Ratio: {perf['Sortino']:.2f}")
    print(f"Calmar Ratio: {perf['Calmar']:.2f}")
    print(f"Max Drawdown: {perf['Max Drawdown']:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve of the strategy vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative Returns'], label=f'{strategy_name} Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} vs Buy and Hold on {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print the current signal and position."""
    df = download_data(ticker, datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    signal = df['Signal'].iloc[-1]
    position = 'Long' if signal == 1 else 'Out'
    print(f"Today's Signal for {ticker}: {position}")

def main():
    parser = argparse.ArgumentParser(description='Adaptive Stop Loss via ATR Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve')
    
    args = parser.parse_args()
    
    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)
    
    if args.plot:
        plot_results(df, args.ticker, 'Adaptive Stop Loss via ATR')
    
    todays_signal(args.ticker)

if __name__ == '__main__':
    main()
```