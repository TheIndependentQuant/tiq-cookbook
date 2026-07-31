```python
# 41 - 200-Day SMA Regime Filter (Bull/Bear)
# Source: The Independent Quant | theindependentquant.com
# 
# The 200-Day Simple Moving Average (SMA) Regime Filter strategy is designed to identify the prevailing market trend for the SPDR S&P 500 ETF Trust (SPY) by examining its price in relation to its 200-day moving average. The strategy generates a bullish signal when the current price of SPY is above its 200-day SMA, indicating a potential upward trend. Conversely, it signals a bearish regime when the price is below the 200-day SMA, suggesting a downward trend or increased volatility. This binary signal helps traders decide whether to enter a long position (when bullish) or stay in cash or short (when bearish).
#
# References:
# (No external references)
#
# Usage instructions:
# Run the script with optional arguments to specify the ticker, start date, end date, and whether to plot the results.
# Example: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Constants
SMA_PERIOD = 200

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the trading signal based on 200-day SMA."""
    df['SMA_200'] = df['Close'].rolling(window=SMA_PERIOD).mean()
    df['Signal'] = np.where(df['Close'] > df['SMA_200'], 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy and compute returns."""
    df['Market_Return'] = df['Close'].pct_change()
    df['Strategy_Return'] = df['Market_Return'] * df['Signal']
    df['Cumulative_Market'] = (1 + df['Market_Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative_Strategy'].iloc[-1]) ** (252/len(df)) - 1
    sharpe = df['Strategy_Return'].mean() / df['Strategy_Return'].std() * np.sqrt(252)
    sortino = df['Strategy_Return'].mean() / df[df['Strategy_Return'] < 0]['Strategy_Return'].std() * np.sqrt(252)
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
    """Plot the equity curve of the strategy versus buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative_Market'], label='Buy and Hold', linestyle='--')
    plt.plot(df.index, df['Cumulative_Strategy'], label=strategy_name)
    plt.title(f'{strategy_name} on {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Print today's signal based on the most recent data."""
    df = download_data(ticker, start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    current_signal = df['Signal'].iloc[-1]
    position = "Bullish" if current_signal == 1 else "Bearish"
    print(f"Today's signal for {ticker}: {position}")

def main():
    """Main function to run the strategy."""
    parser = argparse.ArgumentParser(description="200-Day SMA Regime Filter (Bull/Bear) Strategy")
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', default='2010-01-01', help='Start date (default: 2010-01-01)')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'), help='End date (default: today)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve if set')
    
    args = parser.parse_args()
    
    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)
    
    if args.plot:
        plot_results(df, args.ticker, "200-Day SMA Regime Filter")
    
    todays_signal(args.ticker)

if __name__ == '__main__':
    main()
```