```python
# 12 - Bollinger Band Bounce
# Source: The Independent Quant | theindependentquant.com
# 
# The Bollinger Band Bounce strategy is a mean-reversion trading approach applied to the SPY, an ETF that tracks the S&P 500 index. This strategy seeks to exploit short-term price deviations from a perceived norm, using Bollinger Bands as a key indicator. Bollinger Bands consist of a middle band (a simple moving average) and two outer bands set at a specified number of standard deviations above and below the middle band. In this strategy, the signal is generated when the price of SPY touches or crosses below the lower Bollinger Band, indicating that the asset may be oversold and poised for a bounce back towards the mean, or middle band.
# 
# References:
# (No external references)
# 
# Usage instructions:
# Run the script with optional arguments for ticker, start date, end date, and plot flag.
# Example: python bollinger_band_bounce.py --ticker SPY --start 2010-01-01 --end 2023-01-01 --plot

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Constants
BOLLINGER_PERIOD = 20
BOLLINGER_STD_DEV = 2

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute trading signals based on Bollinger Bands."""
    df['SMA'] = df['Close'].rolling(window=BOLLINGER_PERIOD).mean()
    df['STD'] = df['Close'].rolling(window=BOLLINGER_PERIOD).std()
    df['Lower Band'] = df['SMA'] - (BOLLINGER_STD_DEV * df['STD'])
    
    df['Signal'] = 0
    df.loc[df['Close'] < df['Lower Band'], 'Signal'] = 1
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy and compute returns."""
    df['Strategy Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1])**(252/len(df)) - 1
    returns = df['Strategy Returns'].dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    downside_std = returns[returns < 0].std()
    sortino = returns.mean() / downside_std * np.sqrt(252)
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
        print(f"{key}: {value:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve and buy-and-hold comparison."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label='Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f"{strategy_name} on {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Print today's trading signal based on recent data."""
    end = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - pd.DateOffset(days=30)).strftime('%Y-%m-%d')
    df = download_data(ticker, start, end)
    df = compute_signal(df)
    latest_signal = df['Signal'].iloc[-1]
    position = 'Long' if latest_signal == 1 else 'Neutral'
    print(f"Today's signal for {ticker}: {position}")

def main():
    parser = argparse.ArgumentParser(description='Bollinger Band Bounce Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol')
    parser.add_argument('--start', default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve')
    
    args = parser.parse_args()
    
    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)
    
    if args.plot:
        plot_results(df, args.ticker, "#12 - Bollinger Band Bounce")
    
    todays_signal(args.ticker)

if __name__ == "__main__":
    main()
```