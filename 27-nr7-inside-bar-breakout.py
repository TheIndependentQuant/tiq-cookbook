```python
# 27 - NR7 + Inside Bar Breakout
# Source: The Independent Quant | theindependentquant.com
# The NR7 + Inside Bar Breakout strategy is a systematic trading approach designed to identify potential breakout opportunities in the SPY, an ETF that tracks the S&P 500 index. This strategy combines two specific candlestick patterns: the NR7 (Narrow Range 7) and the Inside Bar. An NR7 day is when the trading range (high minus low) is the smallest it has been over the past seven days, indicating a period of consolidation. An Inside Bar is a day where the entire price range is contained within the high and low of the previous day, suggesting indecision in the market. The strategy predicts that when these two patterns appear consecutively, it signals a potential breakout in either direction. The expectation is that the period of low volatility and consolidation will be followed by a significant price movement. Traders using this strategy look for a breakout above the high or below the low of the Inside Bar to enter a trade, anticipating a continuation of the breakout direction.
# References:
# (No external references)
# Usage instructions:
# Run this script with optional arguments for ticker, start date, end date, and plot flag. For example:
# python nr7_inside_bar.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
NR7_LOOKBACK = 7

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the NR7 + Inside Bar Breakout signal."""
    df['Range'] = df['High'] - df['Low']
    df['NR7'] = df['Range'] == df['Range'].rolling(NR7_LOOKBACK).min()
    df['InsideBar'] = (df['High'] <= df['High'].shift(1)) & (df['Low'] >= df['Low'].shift(1))
    df['Signal'] = np.where(df['NR7'] & df['InsideBar'], 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy and calculate returns."""
    df['StrategyReturns'] = df['Signal'] * df['Close'].pct_change()
    df['CumulativeReturns'] = (1 + df['StrategyReturns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    total_return = df['CumulativeReturns'].iloc[-1] - 1
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1
    sharpe = df['StrategyReturns'].mean() / df['StrategyReturns'].std() * np.sqrt(252)
    downside_std = df['StrategyReturns'][df['StrategyReturns'] < 0].std()
    sortino = df['StrategyReturns'].mean() / downside_std * np.sqrt(252)
    max_drawdown = (df['CumulativeReturns'].cummax() - df['CumulativeReturns']).max()
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
    print(f"Performance for {ticker}:")
    print(f"CAGR: {perf['CAGR']:.2%}")
    print(f"Sharpe Ratio: {perf['Sharpe']:.2f}")
    print(f"Sortino Ratio: {perf['Sortino']:.2f}")
    print(f"Calmar Ratio: {perf['Calmar']:.2f}")
    print(f"Max Drawdown: {perf['Max Drawdown']:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve and compare to buy-and-hold."""
    df['BuyAndHold'] = (1 + df['Close'].pct_change()).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['CumulativeReturns'], label=f'{strategy_name} Strategy')
    plt.plot(df.index, df['BuyAndHold'], label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print current signal and position."""
    df = download_data(ticker, start=datetime.now().strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    signal_today = df['Signal'].iloc[-1]
    print(f"Today's signal for {ticker}: {'Buy' if signal_today == 1 else 'No Signal'}")

def main():
    parser = argparse.ArgumentParser(description='NR7 + Inside Bar Breakout Strategy')
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
        plot_results(df, args.ticker, 'NR7 + Inside Bar Breakout')

    todays_signal(args.ticker)

if __name__ == '__main__':
    main()
```