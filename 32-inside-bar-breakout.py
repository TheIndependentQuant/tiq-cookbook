```python
# 32 - Inside Bar Breakout
# Source: The Independent Quant | theindependentquant.com
# The Inside Bar Breakout strategy is a price action-based approach applied to the SPY ETF, which tracks the S&P 500 index. It identifies potential breakout opportunities by focusing on inside bars, which are candlesticks that form within the high and low range of the previous bar. This pattern suggests a period of consolidation and indecision in the market. The strategy sets buy and sell orders around the high and low of the inside bar, anticipating a breakout in either direction.
# References:
# https://en.wikipedia.org/wiki/Inside_(2023_film)
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and plot flag. Example: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-10 --plot

import pandas as pd
import numpy as np
import yfinance as yf
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
LOOKBACK_PERIOD = 1

def download_data(ticker, start, end):
    """Download historical data for a given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the Inside Bar Breakout signal."""
    df['High_shift'] = df['High'].shift(1)
    df['Low_shift'] = df['Low'].shift(1)
    df['Inside_Bar'] = (df['High'] < df['High_shift']) & (df['Low'] > df['Low_shift'])
    
    df['Signal'] = 0
    df.loc[df['Inside_Bar'] & (df['Close'] > df['High_shift']), 'Signal'] = 1
    df.loc[df['Inside_Bar'] & (df['Close'] < df['Low_shift']), 'Signal'] = -1
    
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy."""
    df['Strategy_Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1]) ** (252/len(df)) - 1
    sharpe = np.sqrt(252) * df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()
    downside_std = df['Strategy_Returns'][df['Strategy_Returns'] < 0].std()
    sortino = np.sqrt(252) * df['Strategy_Returns'].mean() / downside_std
    max_drawdown = (df['Cumulative'].cummax() - df['Cumulative']).max()
    calmar = cagr / max_drawdown
    
    return {
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print the performance results."""
    print(f"Performance Metrics for {ticker}:")
    print(f"CAGR: {perf['CAGR']:.2%}")
    print(f"Sharpe Ratio: {perf['Sharpe']:.2f}")
    print(f"Sortino Ratio: {perf['Sortino']:.2f}")
    print(f"Calmar Ratio: {perf['Calmar']:.2f}")
    print(f"Max Drawdown: {perf['Max Drawdown']:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve and buy-and-hold comparison."""
    df['Buy_and_Hold'] = (1 + df['Close'].pct_change()).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot(df.index, df['Buy_and_Hold'], label='Buy and Hold', linestyle='--')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.grid()
    plt.show()

def todays_signal(ticker):
    """Print today's signal based on recent data."""
    df = download_data(ticker, (datetime.now() - pd.Timedelta(days=30)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    last_signal = df['Signal'].iloc[-1]
    print(f"Today's Signal for {ticker}: {'Buy' if last_signal == 1 else 'Sell' if last_signal == -1 else 'Hold'}")

def main():
    parser = argparse.ArgumentParser(description='Inside Bar Breakout Strategy')
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
        plot_results(df, args.ticker, 'Inside Bar Breakout')
    
    todays_signal(args.ticker)

if __name__ == "__main__":
    main()
```