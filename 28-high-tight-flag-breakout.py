```python
# 28 - High Tight Flag Breakout
# Source: The Independent Quant | theindependentquant.com
# The High Tight Flag Breakout strategy is designed to identify and capitalize on strong upward momentum in the SPDR S&P 500 ETF Trust (SPY). This strategy focuses on spotting a specific chart pattern known as the "high tight flag," which typically indicates a continuation of a bullish trend. The signal is generated when SPY experiences a sharp price increase, followed by a brief consolidation period where the price remains relatively stable within a narrow range. This consolidation forms the "flag" after the initial "flagpole." The breakout signal is triggered when the price moves above the upper boundary of this consolidation range, suggesting a potential continuation of the uptrend.
# References:
# (No external references)
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and plot flag. Example: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Constants
LOOKBACK_PERIOD = 20  # Days for flagpole
CONSOLIDATION_PERIOD = 10  # Days for flag
BREAKOUT_THRESHOLD = 0.02  # 2% breakout threshold
RISK_FREE_RATE = 0.01  # Risk-free rate for performance metrics

def download_data(ticker, start, end):
    """Downloads historical data for a given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Computes the High Tight Flag Breakout signal."""
    df['Flagpole'] = df['Close'].rolling(LOOKBACK_PERIOD).max()
    df['Consolidation'] = df['Close'].rolling(CONSOLIDATION_PERIOD).mean()
    df['Breakout'] = df['Consolidation'] * (1 + BREAKOUT_THRESHOLD)
    df['Signal'] = np.where(df['Close'].shift(1) > df['Breakout'].shift(1), 1, 0)
    return df

def backtest(df):
    """Backtests the strategy and computes returns."""
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculates performance metrics."""
    cagr = (df['Cumulative'].iloc[-1] ** (252.0 / len(df))) - 1
    daily_returns = df['Strategy_Returns'].dropna()
    sharpe_ratio = (daily_returns.mean() - RISK_FREE_RATE / 252) / daily_returns.std() * np.sqrt(252)
    sortino_ratio = (daily_returns.mean() - RISK_FREE_RATE / 252) / daily_returns[daily_returns < 0].std() * np.sqrt(252)
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
    """Prints formatted performance table."""
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plots equity curve vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Downloads recent data and prints current signal and position."""
    df = download_data(ticker, datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    signal = df['Signal'].iloc[-1]
    print(f"Today's signal for {ticker}: {'Buy' if signal else 'Hold'}")

def main():
    """Wires everything together via argparse."""
    parser = argparse.ArgumentParser(description='High Tight Flag Breakout Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol')
    parser.add_argument('--start', default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, 'High Tight Flag Breakout')

if __name__ == '__main__':
    main()
```