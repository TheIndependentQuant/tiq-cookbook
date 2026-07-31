```python
# 33 - Bullish/Bearish Engulfing Candle
# Source: The Independent Quant | theindependentquant.com
# This strategy identifies potential reversals in the SPY ETF by analyzing specific candlestick patterns known as Bullish and Bearish Engulfing patterns. A bullish engulfing pattern occurs when a small bearish candle is followed by a larger bullish candle that completely engulfs the previous day's body, indicating a potential upward reversal. Conversely, a bearish engulfing pattern occurs when a small bullish candle is followed by a larger bearish candle, suggesting a potential downward reversal. These patterns are used to predict market reversals and capitalize on subsequent price movements.
# References:
# https://www.quantum-algo.com/blog/guides/engulfing-candle-complete-guide/
# Usage: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
LOOKBACK_PERIOD = 1

def download_data(ticker='SPY', start='2010-01-01', end=None):
    """Download historical data for the given ticker."""
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute Bullish/Bearish Engulfing Candle signals."""
    df['signal'] = 0
    # Bullish Engulfing
    df.loc[(df['Close'].shift(1) < df['Open'].shift(1)) & 
           (df['Close'] > df['Open']) & 
           (df['Close'] > df['Open'].shift(1)) & 
           (df['Open'] < df['Close'].shift(1)), 'signal'] = 1
    # Bearish Engulfing
    df.loc[(df['Close'].shift(1) > df['Open'].shift(1)) & 
           (df['Close'] < df['Open']) & 
           (df['Close'] < df['Open'].shift(1)) & 
           (df['Open'] > df['Close'].shift(1)), 'signal'] = -1
    df['signal'] = df['signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy based on signals."""
    df['strategy_returns'] = df['signal'] * df['Close'].pct_change()
    df['cumulative'] = (1 + df['strategy_returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['cumulative'].iloc[-1] ** (252 / len(df))) - 1
    volatility = df['strategy_returns'].std() * np.sqrt(252)
    sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
    downside_std = df[df['strategy_returns'] < 0]['strategy_returns'].std() * np.sqrt(252)
    sortino_ratio = df['strategy_returns'].mean() / downside_std
    max_drawdown = (df['cumulative'].cummax() - df['cumulative']).max()
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
    print(f"Performance for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve versus buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['cumulative'], label='Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy & Hold')
    plt.title(f"{strategy_name} Strategy Performance on {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Get today's signal based on the latest data."""
    df = download_data(ticker=ticker, start='2023-01-01')
    df = compute_signal(df)
    latest_signal = df['signal'].iloc[-1]
    position = "Long" if latest_signal == 1 else "Short" if latest_signal == -1 else "Neutral"
    print(f"Today's signal for {ticker}: {position}")

def main():
    parser = argparse.ArgumentParser(description='Bullish/Bearish Engulfing Candle Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve')
    args = parser.parse_args()

    df = download_data(ticker=args.ticker, start=args.start, end=args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, "Bullish/Bearish Engulfing Candle")

if __name__ == "__main__":
    main()
```