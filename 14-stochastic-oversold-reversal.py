```python
# 14 - Stochastic Oversold Reversal
# Source: The Independent Quant | theindependentquant.com
# The Stochastic Oversold Reversal strategy is designed to identify potential buying opportunities in SPY by using the stochastic oscillator. This momentum indicator compares a security's closing price to a range of its prices over a certain period. The strategy generates a buy signal when the stochastic oscillator falls below a threshold (typically 20) and then crosses back above it, suggesting a potential price reversal. This approach aims to capitalize on short-term price corrections by buying SPY when it is undervalued.
# References:
# (No external references)
# Usage: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
STOCHASTIC_PERIOD = 14
OVERSOLD_THRESHOLD = 20

def download_data(ticker, start, end):
    """Download historical data for a given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the Stochastic Oversold Reversal signal."""
    df['Low14'] = df['Low'].rolling(window=STOCHASTIC_PERIOD).min()
    df['High14'] = df['High'].rolling(window=STOCHASTIC_PERIOD).max()
    df['%K'] = 100 * ((df['Close'] - df['Low14']) / (df['High14'] - df['Low14']))
    df['Signal'] = np.where((df['%K'].shift(1) < OVERSOLD_THRESHOLD) & (df['%K'] > OVERSOLD_THRESHOLD), 1, 0)
    return df

def backtest(df):
    """Backtest the strategy."""
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1])**(252/len(df)) - 1
    daily_returns = df['Strategy_Returns'].dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_returns.std()
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
    """Print the performance results."""
    print(f"Performance for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve versus buy-and-hold."""
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
    """Print today's signal for the given ticker."""
    df = download_data(ticker, start='2023-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    latest_signal = df['Signal'].iloc[-1]
    print(f"Today's signal for {ticker} is: {'Buy' if latest_signal == 1 else 'No Action'}")

def main():
    """Main function to wire everything together."""
    parser = argparse.ArgumentParser(description='Stochastic Oversold Reversal Strategy')
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
        plot_results(df, args.ticker, 'Stochastic Oversold Reversal')

if __name__ == "__main__":
    main()
```