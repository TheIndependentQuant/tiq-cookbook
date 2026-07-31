```python
# 11 - RSI(2) Rebound
# Source: The Independent Quant | theindependentquant.com
# The RSI(2) Rebound strategy is a mean-reversion trading system designed to exploit short-term overbought and oversold conditions in the SPY, which is an ETF that tracks the S&P 500 index. The strategy uses the Relative Strength Index (RSI), a momentum oscillator, to identify these conditions. Specifically, it calculates the RSI over a two-day period (RSI(2)), which is a very short timeframe. The RSI(2) value ranges from 0 to 100, and the strategy generates a buy signal when this value falls below a certain threshold, indicating that the SPY is oversold and may rebound.

# References:
# (No external references)

# Usage instructions:
# This script implements the RSI(2) Rebound strategy on SPY. Use the command line arguments to specify the ticker, start and end dates, and whether to plot the results.

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import norm

# Constants
RSI_PERIOD = 2
RSI_THRESHOLD = 10

def download_data(ticker='SPY', start='2010-01-01', end=None):
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start, end=end)
    return data

def compute_signal(df):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).sum()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).sum()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    df['Signal'] = np.where(df['RSI'] < RSI_THRESHOLD, 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    df['Strategy_Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    cagr = (df['Cumulative'].iloc[-1])**(252/len(df)) - 1
    daily_return = df['Strategy_Returns'].mean()
    std_dev = df['Strategy_Returns'].std()
    sharpe_ratio = daily_return / std_dev * np.sqrt(252)
    downside_std = df[df['Strategy_Returns'] < 0]['Strategy_Returns'].std()
    sortino_ratio = daily_return / downside_std * np.sqrt(252)
    max_drawdown = (df['Cumulative'].cummax() - df['Cumulative']).max()
    calmar_ratio = cagr / max_drawdown
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    print(f"Performance for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    df['Buy_and_Hold'] = (1 + df['Close'].pct_change()).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot(df['Buy_and_Hold'], label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold for {ticker}')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    df = download_data(ticker, start=(datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d'))
    df = compute_signal(df)
    last_signal = df['Signal'].iloc[-1]
    print(f"Today's signal for {ticker}: {'Buy' if last_signal == 1 else 'No Action'}")

def main():
    parser = argparse.ArgumentParser(description='RSI(2) Rebound Strategy on SPY')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', default='2010-01-01', help='Start date (default: 2010-01-01)')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'), help='End date (default: today)')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, 'RSI(2) Rebound')

    todays_signal(args.ticker)

if __name__ == '__main__':
    main()
```