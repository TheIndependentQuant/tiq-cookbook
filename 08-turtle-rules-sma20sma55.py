```python
#08 - Turtle Rules (SMA20/SMA55)
# Source: The Independent Quant | theindependentquant.com
# The Turtle Rules (SMA20/SMA55) strategy is a trend-following approach that utilizes simple moving averages to generate trading signals on the SPY ETF. Specifically, it involves calculating two simple moving averages (SMAs): a 20-day SMA and a 55-day SMA. The strategy generates a buy signal when the 20-day SMA crosses above the 55-day SMA, indicating a potential upward trend. Conversely, a sell signal is generated when the 20-day SMA crosses below the 55-day SMA, suggesting a potential downward trend.
# References:
# https://www.mql5.com/en/articles/23155
# Usage: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 55

def download_data(ticker, start, end):
    """Downloads historical data for the specified ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Computes the trading signals based on SMA20/SMA55 crossover."""
    df['SMA20'] = df['Close'].rolling(window=SMA_SHORT_WINDOW).mean()
    df['SMA55'] = df['Close'].rolling(window=SMA_LONG_WINDOW).mean()
    df['Signal'] = 0
    df['Signal'][SMA_LONG_WINDOW:] = np.where(df['SMA20'][SMA_LONG_WINDOW:] > df['SMA55'][SMA_LONG_WINDOW:], 1, -1)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtests the strategy and calculates strategy returns."""
    df['Returns'] = df['Close'].pct_change()
    df['Strategy'] = df['Signal'] * df['Returns']
    df['Cumulative'] = (1 + df['Strategy']).cumprod()
    return df

def performance(df):
    """Calculates performance metrics for the strategy."""
    cagr = (df['Cumulative'].iloc[-1])**(252/len(df)) - 1
    sharpe_ratio = (df['Strategy'].mean() / df['Strategy'].std()) * np.sqrt(252)
    downside_std = df[df['Strategy'] < 0]['Strategy'].std()
    sortino_ratio = (df['Strategy'].mean() / downside_std) * np.sqrt(252)
    max_drawdown = (df['Cumulative'].cummax() - df['Cumulative']).max()
    calmar_ratio = cagr / max_drawdown if max_drawdown != 0 else np.nan
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Prints the performance metrics in a formatted table."""
    print(f"Performance Metrics for {ticker}")
    print(f"{'Metric':<20}{'Value':<20}")
    for key, value in perf.items():
        print(f"{key:<20}{value:<20.2f}")

def plot_results(df, ticker, strategy_name):
    """Plots the equity curve of the strategy versus buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot((1 + df['Returns']).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold on {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Prints the current signal and position based on recent data."""
    df = download_data(ticker, start='2022-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    current_signal = df['Signal'].iloc[-1]
    current_position = 'Long' if current_signal == 1 else 'Short'
    print(f"Today's Signal for {ticker}: {current_position}")

def main():
    parser = argparse.ArgumentParser(description='Turtle Rules (SMA20/SMA55) Strategy')
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
        plot_results(df, args.ticker, 'Turtle Rules (SMA20/SMA55)')

    todays_signal(args.ticker)

if __name__ == '__main__':
    main()
```