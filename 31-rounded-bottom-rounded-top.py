```python
# 31 - Rounded Bottom / Rounded Top
# Source: The Independent Quant | theindependentquant.com
# The Rounded Bottom / Rounded Top strategy is a price action-based approach designed to identify potential reversals in the SPY, the ETF that tracks the S&P 500 index. This strategy seeks to capture turning points in the market by identifying patterns where prices gradually form a rounded shape at the bottom or top of a trend. The signal is computed by analyzing the curvature of the price action over a specified period, typically using a simple moving average to smooth out price fluctuations and highlight the rounded formation. A rounded bottom indicates a potential bullish reversal, suggesting that prices may start to rise, while a rounded top suggests a bearish reversal, indicating a potential decline.
# References:
# (No external references)
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and plot flag to visualize results.

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
LOOKBACK_PERIOD = 20
THRESHOLD = 0.01  # Threshold for curvature

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    df['SMA'] = df['Adj Close'].rolling(window=LOOKBACK_PERIOD).mean()
    return df

def compute_signal(df):
    """Compute trading signals based on rounded bottom/top patterns."""
    df['Curvature'] = df['SMA'].diff().diff()
    df['Signal'] = 0
    df.loc[df['Curvature'] > THRESHOLD, 'Signal'] = 1  # Rounded Bottom
    df.loc[df['Curvature'] < -THRESHOLD, 'Signal'] = -1  # Rounded Top
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy based on signals."""
    df['Strategy_Returns'] = df['Signal'] * df['Adj Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1])**(252/len(df)) - 1
    daily_returns = df['Strategy_Returns'].dropna()
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    sortino = np.sqrt(252) * daily_returns.mean() / daily_returns[daily_returns < 0].std()
    max_drawdown = (df['Cumulative'].cummax() - df['Cumulative']).max()
    calmar = cagr / max_drawdown
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print the performance results."""
    print(f"Performance for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve of the strategy vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label='Strategy')
    plt.plot((1 + df['Adj Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} on {ticker}')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print the current signal."""
    df = download_data(ticker, start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    print(f"Today's signal for {ticker}: {df['Signal'].iloc[-1]}")

def main():
    parser = argparse.ArgumentParser(description='Rounded Bottom / Rounded Top Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', default='2010-01-01', help='Start date (default: 2010-01-01)')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help='End date (default: today)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve if set')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, "#31 - Rounded Bottom / Rounded Top")

if __name__ == '__main__':
    main()
```