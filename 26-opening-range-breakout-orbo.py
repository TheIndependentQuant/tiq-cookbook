```python
# 26 - Opening Range Breakout (ORBO)
# Source: The Independent Quant | theindependentquant.com
# The Opening Range Breakout (ORBO) strategy focuses on the SPY, an ETF that tracks the S&P 500 index. This strategy identifies a specific time window at the beginning of the trading day, typically the first 30 to 60 minutes, to establish an "opening range." The high and low prices during this period set the boundaries for potential breakouts. When the SPY price moves above the opening range high, a buy signal is generated, predicting a continuation of bullish momentum. Conversely, if the price falls below the opening range low, it signals a potential bearish trend, suggesting a short position.
# References:
# https://www.merriam-webster.com/dictionary/opening
# Usage: python orbo_strategy.py --ticker SPY --start 2010-01-01 --end 2023-10-10 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
OPENING_RANGE_MINUTES = 30

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end, interval='1d')
    return df

def compute_signal(df):
    """Compute the ORBO signal."""
    df['Opening Range High'] = df['High'].rolling(window=OPENING_RANGE_MINUTES).max()
    df['Opening Range Low'] = df['Low'].rolling(window=OPENING_RANGE_MINUTES).min()
    df['Signal'] = 0
    df.loc[df['Close'] > df['Opening Range High'].shift(1), 'Signal'] = 1
    df.loc[df['Close'] < df['Opening Range Low'].shift(1), 'Signal'] = -1
    return df

def backtest(df):
    """Backtest the ORBO strategy."""
    df['Strategy Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
    df['Cumulative Returns'] = (1 + df['Strategy Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative Returns'].iloc[-1]) ** (252 / len(df)) - 1
    sharpe = np.sqrt(252) * df['Strategy Returns'].mean() / df['Strategy Returns'].std()
    downside_std = df[df['Strategy Returns'] < 0]['Strategy Returns'].std()
    sortino = np.sqrt(252) * df['Strategy Returns'].mean() / downside_std
    df['Drawdown'] = df['Cumulative Returns'] / df['Cumulative Returns'].cummax() - 1
    max_drawdown = df['Drawdown'].min()
    calmar = cagr / abs(max_drawdown)
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
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve and buy-and-hold comparison."""
    df['Buy and Hold'] = (1 + df['Close'].pct_change()).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative Returns'], label=f'{strategy_name} Strategy')
    plt.plot(df['Buy and Hold'], label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold for {ticker}')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Print today's signal based on recent data."""
    today = datetime.now().strftime('%Y-%m-%d')
    df = download_data(ticker, start=today, end=today)
    df = compute_signal(df)
    signal = df['Signal'].iloc[-1]
    position = 'Long' if signal == 1 else 'Short' if signal == -1 else 'Neutral'
    print(f"Today's Signal for {ticker}: {position}")

def main():
    parser = argparse.ArgumentParser(description='Opening Range Breakout (ORBO) Strategy')
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
        plot_results(df, args.ticker, 'Opening Range Breakout (ORBO)')

if __name__ == '__main__':
    main()
```