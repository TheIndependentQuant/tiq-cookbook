```python
# 40 - Outside Bar Continuation
# Source: The Independent Quant | theindependentquant.com
# The Outside Bar Continuation strategy is a price action-based trading approach that focuses on identifying and capitalizing on specific candlestick patterns within the SPY ETF, which tracks the S&P 500 index. An "outside bar" occurs when the high and low of a given candlestick completely engulf the high and low of the previous candlestick. This pattern suggests a potential continuation of the current trend, whether bullish or bearish. The strategy aims to predict short-term price movements by entering trades in the direction of the trend following the appearance of an outside bar.
# References:
# (No external references)
# Usage instructions: Run the script with optional arguments for ticker, start and end dates, and plot flag to see the equity curve.

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
LOOKBACK_PERIOD = 1

def download_data(ticker='SPY', start='2010-01-01', end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)
    df['outside_bar'] = ((df['High'] > df['prev_high']) & (df['Low'] < df['prev_low'])).astype(int)
    df['signal'] = df['outside_bar'].shift(1)
    return df

def backtest(df):
    df['strategy_returns'] = df['signal'] * df['Close'].pct_change()
    df['cumulative'] = (1 + df['strategy_returns']).cumprod()
    return df

def performance(df):
    cagr = (df['cumulative'].iloc[-1])**(252/len(df)) - 1
    sharpe = np.mean(df['strategy_returns']) / np.std(df['strategy_returns']) * np.sqrt(252)
    downside_std = np.std(df['strategy_returns'][df['strategy_returns'] < 0])
    sortino = np.mean(df['strategy_returns']) / downside_std * np.sqrt(252)
    max_drawdown = (df['cumulative'].cummax() - df['cumulative']).max()
    calmar = cagr / max_drawdown
    return {
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    print(f"Performance for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2%}")

def plot_results(df, ticker, strategy_name):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['cumulative'], label='Strategy')
    plt.plot(df.index, (1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f"{strategy_name} on {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker='SPY'):
    df = download_data(ticker, start=(datetime.today().strftime('%Y-%m-%d')))
    df = compute_signal(df)
    signal = df['signal'].iloc[-1]
    print(f"Today's signal for {ticker}: {'Long' if signal == 1 else 'No Position'}")

def main():
    parser = argparse.ArgumentParser(description='Outside Bar Continuation Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol')
    parser.add_argument('--start', default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, "Outside Bar Continuation")

if __name__ == "__main__":
    main()
```