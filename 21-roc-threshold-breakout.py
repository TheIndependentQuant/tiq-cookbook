```python
# 21 - ROC Threshold Breakout
# Source: The Independent Quant | theindependentquant.com
# The ROC Threshold Breakout strategy is a momentum-based trading approach specifically designed for the SPY, which is the ETF tracking the S&P 500 index. The strategy utilizes the Rate of Change (ROC) indicator to identify potential breakout opportunities. The ROC is a momentum oscillator that measures the percentage change in price between the current price and the price a certain number of periods ago. In this strategy, a 20-day ROC is calculated to assess the momentum of SPY. When the ROC exceeds a predetermined threshold, it signals a potential breakout, suggesting that the price of SPY is likely to continue moving in the same direction.
# References:
# https://www.rocskincare.com/
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and plot flag to visualize results.

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Constants
ROC_PERIOD = 20
ROC_THRESHOLD = 5.0

def download_data(ticker, start, end):
    """Download historical data for a given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the ROC signal for the strategy."""
    df['ROC'] = (df['Adj Close'].pct_change(ROC_PERIOD) * 100)
    df['Signal'] = np.where(df['ROC'] > ROC_THRESHOLD, 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy and compute returns."""
    df['Strategy_Returns'] = df['Signal'] * df['Adj Close'].pct_change()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    df['Cumulative_Buy_Hold'] = (1 + df['Adj Close'].pct_change()).cumprod()
    return df

def performance(df):
    """Calculate performance metrics for the strategy."""
    cagr = (df['Cumulative_Strategy'].iloc[-1])**(252/len(df)) - 1
    sharpe = np.sqrt(252) * df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()
    downside_std = df[df['Strategy_Returns'] < 0]['Strategy_Returns'].std()
    sortino = np.sqrt(252) * df['Strategy_Returns'].mean() / downside_std
    max_drawdown = (df['Cumulative_Strategy'].cummax() - df['Cumulative_Strategy']).max()
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
    plt.plot(df['Cumulative_Strategy'], label=f'{strategy_name} Strategy')
    plt.plot(df['Cumulative_Buy_Hold'], label='Buy & Hold')
    plt.title(f'{strategy_name} Strategy vs Buy & Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print the current signal and position."""
    df = download_data(ticker, start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    print(f"Today's signal for {ticker}: {df['Signal'].iloc[-1]}")

def main():
    """Main function to wire everything together."""
    parser = argparse.ArgumentParser(description='ROC Threshold Breakout Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot the results')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, 'ROC Threshold Breakout')

if __name__ == '__main__':
    main()
```