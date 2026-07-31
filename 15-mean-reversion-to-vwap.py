```python
# 15 - Mean Reversion to VWAP
# Source: The Independent Quant | theindependentquant.com
# The Mean Reversion to VWAP strategy is a systematic trading approach focused on the SPY ETF, which tracks the S&P 500 index. This strategy aims to exploit short-term price deviations from the Volume Weighted Average Price (VWAP). VWAP is a trading benchmark that gives the average price a security has traded at throughout the day, based on both volume and price. The strategy generates a buy signal when the price of SPY falls significantly below its VWAP, suggesting an oversold condition, and predicts that the price will revert back to the VWAP level.
# References:
# (No external references)
# Usage instructions: Run this script with optional arguments for ticker, start, end dates, and plot flag.

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
VWAP_LOOKBACK_PERIOD = 5  # days
PRICE_VWAP_THRESHOLD = -0.02  # 2% below VWAP

def download_data(ticker='SPY', start='2010-01-01', end=None):
    """Download historical data for the given ticker."""
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the trading signal based on mean reversion to VWAP."""
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    df['Price_VWAP_Diff'] = (df['Close'] - df['VWAP']) / df['VWAP']
    df['Signal'] = np.where(df['Price_VWAP_Diff'].shift(1) < PRICE_VWAP_THRESHOLD, 1, 0)
    return df

def backtest(df):
    """Backtest the strategy and calculate returns."""
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative_Returns'].iloc[-1])**(252/len(df)) - 1
    sharpe = np.mean(df['Strategy_Returns']) / np.std(df['Strategy_Returns']) * np.sqrt(252)
    downside_std = np.std(df['Strategy_Returns'][df['Strategy_Returns'] < 0])
    sortino = np.mean(df['Strategy_Returns']) / downside_std * np.sqrt(252)
    max_drawdown = ((df['Cumulative_Returns'].cummax() - df['Cumulative_Returns']) / df['Cumulative_Returns'].cummax()).max()
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
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve of the strategy."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative_Returns'], label=f'{strategy_name} Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print the current signal and position."""
    df = download_data(ticker, start=datetime.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    print(f"Today's signal for {ticker}: {df['Signal'].iloc[-1]}")

def main():
    """Main function to parse arguments and run the strategy."""
    parser = argparse.ArgumentParser(description='Mean Reversion to VWAP Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol for the stock/ETF')
    parser.add_argument('--start', default='2010-01-01', help='Start date for historical data')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help='End date for historical data')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve if set')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)
    
    if args.plot:
        plot_results(df, args.ticker, "Mean Reversion to VWAP")
    
    todays_signal(args.ticker)

if __name__ == '__main__':
    main()
```