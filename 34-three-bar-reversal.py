```python
# 34 - Three-Bar Reversal
# Source: The Independent Quant | theindependentquant.com
# The Three-Bar Reversal strategy is a price action trading method designed to identify potential reversals in the SPY, an ETF that tracks the S&P 500 index. The strategy looks for a specific pattern over three consecutive trading days. The signal is generated when the following conditions are met: on the first day, the market experiences a down day; on the second day, the market continues to decline, closing lower than the first day; and on the third day, the market reverses, closing higher than the second day's high. This pattern suggests a potential bullish reversal, indicating that the SPY might move upward in the near term.
# References:
# https://www.finanztip.de/kfz-versicherung/fuer-rentner/
# Usage: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Constants
LOOKBACK_DAYS = 3

def download_data(ticker, start, end):
    """Download historical data for a given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the Three-Bar Reversal signal."""
    df['Signal'] = 0
    df['Prev_Close'] = df['Close'].shift(1)
    df['Prev2_Close'] = df['Close'].shift(2)
    df['Prev2_High'] = df['High'].shift(2)
    
    df.loc[(df['Close'] > df['Prev2_High']) & 
           (df['Prev_Close'] < df['Prev2_Close']) & 
           (df['Prev2_Close'] < df['Prev2_Close'].shift(1)), 'Signal'] = 1
    
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy and calculate returns."""
    df['Strategy_Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    df['Buy_Hold'] = (1 + df['Close'].pct_change()).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1])**(252/len(df)) - 1
    daily_returns = df['Strategy_Returns']
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    downside_std = daily_returns[daily_returns < 0].std()
    sortino = np.sqrt(252) * daily_returns.mean() / downside_std
    max_drawdown = (df['Cumulative'].cummax() - df['Cumulative']).max()
    calmar = cagr / max_drawdown if max_drawdown != 0 else np.nan

    return {
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print formatted performance table."""
    print(f"Performance Metrics for {ticker}:")
    print(f"CAGR: {perf['CAGR']:.2%}")
    print(f"Sharpe Ratio: {perf['Sharpe']:.2f}")
    print(f"Sortino Ratio: {perf['Sortino']:.2f}")
    print(f"Calmar Ratio: {perf['Calmar']:.2f}")
    print(f"Max Drawdown: {perf['Max Drawdown']:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot equity curve vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot(df.index, df['Buy_Hold'], label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold on {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print current signal and position."""
    df = download_data(ticker, datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    current_signal = df['Signal'].iloc[-1]
    print(f"Today's signal for {ticker}: {'Buy' if current_signal == 1 else 'No Signal'}")

def main():
    parser = argparse.ArgumentParser(description='Three-Bar Reversal Strategy')
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
        plot_results(df, args.ticker, 'Three-Bar Reversal')

if __name__ == "__main__":
    main()
```