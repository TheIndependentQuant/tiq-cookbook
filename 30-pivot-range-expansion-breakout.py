```python
# 30 - Pivot Range Expansion Breakout
# Source: The Independent Quant | theindependentquant.com
# The Pivot Range Expansion Breakout strategy is designed to capitalize on breakout movements in SPY, the ETF tracking the S&P 500 index. By using pivot points calculated from the previous day's high, low, and close prices, the strategy identifies potential bullish or bearish trends when the SPY price breaks out above or below a defined pivot range. This strategy aims to capture short-term price movements by entering trades in the direction of the breakout, leveraging SPY's volatility and liquidity.
# References:
# (No external references)
# Usage instructions:
# Run the script with optional arguments to specify the ticker, start and end dates, and whether to plot the results. Example: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-10 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
LOOKBACK_PERIOD = 1

def download_data(ticker, start, end):
    """Downloads historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Computes the Pivot Range Expansion Breakout signal."""
    df['PP'] = (df['High'].shift(LOOKBACK_PERIOD) + df['Low'].shift(LOOKBACK_PERIOD) + df['Close'].shift(LOOKBACK_PERIOD)) / 3
    df['R1'] = 2 * df['PP'] - df['Low'].shift(LOOKBACK_PERIOD)
    df['S1'] = 2 * df['PP'] - df['High'].shift(LOOKBACK_PERIOD)
    
    df['signal'] = 0
    df.loc[df['Close'] > df['R1'], 'signal'] = 1
    df.loc[df['Close'] < df['S1'], 'signal'] = -1
    
    df['signal'] = df['signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtests the strategy and calculates returns."""
    df['strategy_returns'] = df['signal'] * df['Close'].pct_change()
    df['cumulative'] = (1 + df['strategy_returns']).cumprod()
    return df

def performance(df):
    """Calculates performance metrics."""
    cagr = (df['cumulative'].iloc[-1] ** (1 / ((df.index[-1] - df.index[0]).days / 365.25))) - 1
    sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
    sortino = df['strategy_returns'].mean() / df[df['strategy_returns'] < 0]['strategy_returns'].std() * np.sqrt(252)
    max_drawdown = (df['cumulative'].cummax() - df['cumulative']).max()
    calmar = cagr / max_drawdown if max_drawdown != 0 else np.nan
    
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Prints the performance metrics."""
    print(f"Performance for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plots the equity curve of the strategy vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['cumulative'], label='Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} on {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Prints today's signal for the given ticker."""
    df = download_data(ticker, datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    signal = df['signal'].iloc[-1]
    position = 'Long' if signal == 1 else 'Short' if signal == -1 else 'Neutral'
    print(f"Today's signal for {ticker}: {position}")

def main():
    parser = argparse.ArgumentParser(description='Pivot Range Expansion Breakout Strategy')
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
        plot_results(df, args.ticker, "Pivot Range Expansion Breakout")

if __name__ == '__main__':
    main()
```