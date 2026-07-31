```python
# 39 - Gap Fill Mean Reversion
# Source: The Independent Quant | theindependentquant.com
# 
# This strategy focuses on capitalizing on price gaps in the SPDR S&P 500 ETF Trust (SPY), which tracks the S&P 500 index. 
# A price gap occurs when the opening price of SPY is significantly different from its previous closing price. 
# The strategy predicts that such gaps will tend to close, or "fill," over the course of the trading day. 
# Specifically, if SPY opens significantly lower than its previous close, the strategy anticipates a mean reversion where 
# the price will rise to close the gap. Conversely, if SPY opens significantly higher, it expects the price to fall back 
# toward the previous close. The signal for this strategy is computed by measuring the percentage difference between the 
# opening price and the previous day's closing price. A large gap, either positive or negative, triggers a trade in the 
# opposite direction, betting on the gap closing.
#
# References:
# (No external references)
#
# Usage instructions:
# Run the script in a Python environment. Use the --ticker, --start, --end, and --plot arguments to customize the execution.
# Example: python gap_fill_mean_reversion.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
GAP_THRESHOLD = 0.005  # 0.5% gap
RISK_FREE_RATE = 0.01  # 1% annual risk-free rate

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the trading signal based on gap fill mean reversion."""
    df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['Signal'] = np.where(df['Gap'] > GAP_THRESHOLD, -1, np.where(df['Gap'] < -GAP_THRESHOLD, 1, 0))
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy and compute returns."""
    df['Strategy_Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1] ** (252 / len(df))) - 1
    daily_returns = df['Strategy_Returns'].dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    downside_std = daily_returns[daily_returns < 0].std()
    sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(252)
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
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve of the strategy versus buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label='Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f"{strategy_name} on {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Print today's trading signal based on the most recent data."""
    df = download_data(ticker, start='2023-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    current_signal = df['Signal'].iloc[-1]
    print(f"Today's Signal for {ticker}: {'Long' if current_signal == 1 else 'Short' if current_signal == -1 else 'No Position'}")

def main():
    """Main function to wire everything together."""
    parser = argparse.ArgumentParser(description='Gap Fill Mean Reversion Strategy')
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
        plot_results(df, args.ticker, "Gap Fill Mean Reversion")

if __name__ == "__main__":
    main()
```