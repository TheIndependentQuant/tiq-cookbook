```python
# 44 - SMA Angle Regime Classification
# Source: The Independent Quant | theindependentquant.com
# The SMA Angle Regime Classification strategy predicts future price movements of SPY by analyzing the angle of its Simple Moving Average (SMA). This angle, calculated as the arctangent of the SMA's slope, indicates the market regime: a positive angle suggests an upward trend, while a negative angle indicates a downward trend. The strategy generates buy signals when the angle exceeds a positive threshold and sell signals when it falls below a negative threshold, aiming to profit from market momentum.
# References:
# (No external references)
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and a plot flag. Example: python script.py --ticker SPY --start 2010-01-01 --end 2023-01-01 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from math import atan, degrees

# Constants
SMA_PERIOD = 20
ANGLE_THRESHOLD = 5  # degrees

def download_data(ticker, start, end):
    """Download historical data for a given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute SMA angle and generate trading signals."""
    df['SMA'] = df['Close'].rolling(SMA_PERIOD).mean()
    df['SMA_Slope'] = df['SMA'].diff()
    df['SMA_Angle'] = df['SMA_Slope'].apply(lambda x: degrees(atan(x)))
    df['Signal'] = 0
    df.loc[df['SMA_Angle'] > ANGLE_THRESHOLD, 'Signal'] = 1
    df.loc[df['SMA_Angle'] < -ANGLE_THRESHOLD, 'Signal'] = -1
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Perform backtest on the generated signals."""
    df['Market_Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Signal'] * df['Market_Returns']
    df['Cumulative_Market'] = (1 + df['Market_Returns']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative_Strategy'].iloc[-1]) ** (252 / len(df)) - 1
    sharpe = df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * np.sqrt(252)
    sortino = df['Strategy_Returns'].mean() / df[df['Strategy_Returns'] < 0]['Strategy_Returns'].std() * np.sqrt(252)
    max_drawdown = ((df['Cumulative_Strategy'].cummax() - df['Cumulative_Strategy']).max())
    calmar = cagr / max_drawdown
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar
    }

def print_results(perf, ticker):
    """Print formatted performance table."""
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot equity curve vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative_Market'], label='Buy and Hold', linestyle='--')
    plt.plot(df['Cumulative_Strategy'], label=strategy_name)
    plt.title(f"{strategy_name} vs Buy and Hold on {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print current signal and position."""
    today = datetime.now().strftime('%Y-%m-%d')
    df = download_data(ticker, start='2023-01-01', end=today)
    df = compute_signal(df)
    print(f"Today's signal for {ticker}: {df['Signal'].iloc[-1]}")

def main():
    """Wire everything together via argparse."""
    parser = argparse.ArgumentParser(description='SMA Angle Regime Classification Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', default='2010-01-01', help='Start date (default: 2010-01-01)')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'), help='End date (default: today)')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve chart if set')

    args = parser.parse_args()
    
    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, "SMA Angle Regime Classification")

if __name__ == "__main__":
    main()
```