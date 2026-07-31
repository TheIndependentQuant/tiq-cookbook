```python
# 43 - ADX < 15 Mean Reversion On
# Source: The Independent Quant | theindependentquant.com
# Description: This strategy exploits periods of low market volatility in SPY by using the Average Directional Index (ADX) to identify weak trends. When the ADX falls below 15, indicating a sideways market, the strategy predicts mean reversion. It aims to buy low and sell high during these periods, capitalizing on price oscillations around the mean.
# References:
# https://www.adx.faa.gov/portal/
# Usage: Run the script with optional arguments for ticker, start date, end date, and plot flag.

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
ADX_THRESHOLD = 15
ADX_PERIOD = 14

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the ADX < 15 Mean Reversion On signal."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    df['TR'] = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    df['ATR'] = df['TR'].rolling(window=ADX_PERIOD).mean()
    
    df['+DM'] = np.where((high - high.shift(1)) > (low.shift(1) - low), high - high.shift(1), 0)
    df['-DM'] = np.where((low.shift(1) - low) > (high - high.shift(1)), low.shift(1) - low, 0)
    
    df['+DI'] = 100 * (df['+DM'] / df['ATR']).rolling(window=ADX_PERIOD).mean()
    df['-DI'] = 100 * (df['-DM'] / df['ATR']).rolling(window=ADX_PERIOD).mean()
    
    df['DX'] = (abs(df['+DI'] - df['-DI']) / abs(df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].rolling(window=ADX_PERIOD).mean()
    
    df['Signal'] = np.where(df['ADX'] < ADX_THRESHOLD, 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy."""
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Signal'] * df['Returns']
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1]) ** (252/len(df)) - 1
    sharpe = np.sqrt(252) * df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()
    downside_std = df[df['Strategy_Returns'] < 0]['Strategy_Returns'].std()
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
    """Print formatted performance table."""
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot equity curve vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot((1 + df['Returns']).cumprod(), label='Buy and Hold')
    plt.title(f'Equity Curve for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print current signal and position."""
    df = download_data(ticker, datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    signal = df['Signal'].iloc[-1]
    print(f"Today's Signal for {ticker}: {'Buy' if signal == 1 else 'No Position'}")

def main():
    parser = argparse.ArgumentParser(description='ADX < 15 Mean Reversion On Strategy')
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
        plot_results(df, args.ticker, 'ADX < 15 Mean Reversion On')
    
    todays_signal(args.ticker)

if __name__ == '__main__':
    main()
```