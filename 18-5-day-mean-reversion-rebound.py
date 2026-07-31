```python
# 18 - 5-Day Mean Reversion Rebound
# Source: The Independent Quant | theindependentquant.com
# 
# The 5-Day Mean Reversion Rebound strategy is designed to capitalize on short-term price reversals in the SPY, which is an ETF that tracks the S&P 500 index. This strategy identifies potential buying opportunities when the SPY experiences a significant short-term decline. The exact signal is computed by analyzing the Relative Strength Index (RSI) over a 5-day period. Specifically, the strategy looks for instances where the RSI falls below a certain threshold, indicating that the SPY may be oversold and poised for a rebound.
#
# References:
# (No external references)
#
# Usage instructions:
# Run the script with optional arguments for ticker, start date, end date, and plot flag.
# Example: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
RSI_PERIOD = 5
RSI_THRESHOLD = 30

def download_data(ticker, start, end):
    """Downloads historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Computes the 5-Day Mean Reversion Rebound signal."""
    df['Change'] = df['Adj Close'].diff()
    df['Gain'] = np.where(df['Change'] > 0, df['Change'], 0)
    df['Loss'] = np.where(df['Change'] < 0, -df['Change'], 0)
    df['Avg Gain'] = df['Gain'].rolling(window=RSI_PERIOD).mean()
    df['Avg Loss'] = df['Loss'].rolling(window=RSI_PERIOD).mean()
    df['RS'] = df['Avg Gain'] / df['Avg Loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    df['Signal'] = np.where(df['RSI'] < RSI_THRESHOLD, 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtests the strategy and computes strategy returns."""
    df['Strategy Returns'] = df['Signal'] * df['Adj Close'].pct_change()
    df['Cumulative Strategy Returns'] = (1 + df['Strategy Returns']).cumprod()
    df['Cumulative Market Returns'] = (1 + df['Adj Close'].pct_change()).cumprod()
    return df

def performance(df):
    """Calculates performance metrics."""
    total_days = (df.index[-1] - df.index[0]).days
    cagr = (df['Cumulative Strategy Returns'].iloc[-1] ** (365.0 / total_days)) - 1
    daily_return = df['Strategy Returns'].mean()
    daily_volatility = df['Strategy Returns'].std()
    sharpe_ratio = (daily_return / daily_volatility) * np.sqrt(252)
    downside_volatility = df[df['Strategy Returns'] < 0]['Strategy Returns'].std()
    sortino_ratio = (daily_return / downside_volatility) * np.sqrt(252)
    max_drawdown = ((df['Cumulative Strategy Returns'].cummax() - df['Cumulative Strategy Returns']).max())
    calmar_ratio = cagr / max_drawdown
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Prints the performance results."""
    print(f"Performance Metrics for {ticker}:")
    print(f"CAGR: {perf['CAGR']:.2%}")
    print(f"Sharpe Ratio: {perf['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio: {perf['Sortino Ratio']:.2f}")
    print(f"Calmar Ratio: {perf['Calmar Ratio']:.2f}")
    print(f"Max Drawdown: {perf['Max Drawdown']:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plots the equity curve of the strategy versus buy-and-hold."""
    plt.figure(figsize=(14, 7))
    plt.plot(df['Cumulative Strategy Returns'], label=f'{strategy_name} Strategy')
    plt.plot(df['Cumulative Market Returns'], label='Buy and Hold')
    plt.title(f'{strategy_name} vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Prints today's signal for the given ticker."""
    df = download_data(ticker, start='2023-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    current_signal = df['Signal'].iloc[-1]
    print(f"Today's Signal for {ticker}: {'Buy' if current_signal == 1 else 'Hold'}")

def main():
    """Main function to parse arguments and run the strategy."""
    parser = argparse.ArgumentParser(description='5-Day Mean Reversion Rebound Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Stock ticker (default: SPY)')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (default: 2010-01-01)')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (default: today)')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve if set')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, '5-Day Mean Reversion Rebound')

    todays_signal(args.ticker)

if __name__ == '__main__':
    main()
```