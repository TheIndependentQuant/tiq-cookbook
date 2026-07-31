```python
# 36 - Break and Retest
# Source: The Independent Quant | theindependentquant.com
# The Break and Retest strategy is a systematic trading approach focused on the SPY, an ETF that tracks the S&P 500 index. 
# This strategy identifies potential trading opportunities by analyzing price action patterns. Specifically, it looks for instances 
# where the price breaks above a defined resistance level and then retests this level as a new support. The signal is generated 
# when the SPY's price breaks above a recent high and subsequently pulls back to test that high as a support level without falling 
# below it. This pattern suggests a continuation of the upward trend, predicting a bullish move in SPY.
# References: (No external references)
# Usage instructions: Run the script with default parameters or specify --ticker, --start, --end, and --plot to customize.

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Constants
LOOKBACK_PERIOD = 20  # Days to look back for high
RETEST_TOLERANCE = 0.005  # 0.5% tolerance for retest
RISK_FREE_RATE = 0.01  # Annual risk-free rate for Sharpe/Sortino

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the Break and Retest signal."""
    df['Recent_High'] = df['High'].rolling(window=LOOKBACK_PERIOD).max()
    df['Signal'] = 0
    df['Signal'] = np.where((df['Close'] > df['Recent_High'].shift(1)) &
                            (df['Low'] <= df['Recent_High'].shift(1) * (1 + RETEST_TOLERANCE)) &
                            (df['Low'] >= df['Recent_High'].shift(1) * (1 - RETEST_TOLERANCE)), 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy."""
    df['Strategy_Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    df['Cumulative_Buy_Hold'] = (1 + df['Close'].pct_change()).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative_Strategy'].iloc[-1]) ** (252 / len(df)) - 1
    daily_returns = df['Strategy_Returns'].dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    downside_std = daily_returns[daily_returns < 0].std()
    sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(252)
    max_drawdown = ((df['Cumulative_Strategy'] / df['Cumulative_Strategy'].cummax()) - 1).min()
    calmar_ratio = cagr / abs(max_drawdown)
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print formatted performance table."""
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot equity curve vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative_Strategy'], label='Strategy')
    plt.plot(df['Cumulative_Buy_Hold'], label='Buy and Hold')
    plt.title(f"{strategy_name} on {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print current signal and position."""
    df = download_data(ticker, start=datetime.now().strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    current_signal = df['Signal'].iloc[-1]
    print(f"Today's Signal for {ticker}: {'Buy' if current_signal == 1 else 'No Position'}")

def main():
    parser = argparse.ArgumentParser(description='Break and Retest Strategy')
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
        plot_results(df, args.ticker, "Break and Retest Strategy")

if __name__ == "__main__":
    main()
```