```python
# 46 - Time-Based Exit Strategy (4-Bar Exit)
# Source: The Independent Quant | theindependentquant.com
# 
# The Time-Based Exit Strategy, specifically the 4-Bar Exit, is a systematic trading approach designed for the SPY ETF, which tracks the S&P 500 index. The strategy generates a buy signal when specific conditions are met, and it holds the position for a predetermined number of bars or trading days, in this case, four. The exact signal involves entering a position based on a short-term trend or momentum indicator, such as the Relative Strength Index (RSI), which helps identify overbought or oversold conditions. Once a position is initiated, the strategy holds it for exactly four trading days before exiting, regardless of the price movement during that period.
#
# References:
# (No external references)
#
# Usage instructions:
# Run the script with optional arguments --ticker, --start, --end, and --plot to customize the backtest and view results.

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import math

# Constants
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
HOLD_PERIOD = 4

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the trading signal based on RSI and 4-bar exit."""
    df['RSI'] = compute_rsi(df['Close'], RSI_PERIOD)
    df['signal'] = np.where(df['RSI'] < RSI_OVERSOLD, 1, 0)
    df['signal'] = df['signal'].shift(1)  # Prevent look-ahead bias
    df['position'] = df['signal'].rolling(window=HOLD_PERIOD).sum().shift(1)
    df['position'] = np.where(df['position'] > 0, 1, 0)
    return df

def compute_rsi(series, period):
    """Compute the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest(df):
    """Backtest the strategy and compute returns."""
    df['strategy_returns'] = df['position'] * df['Close'].pct_change()
    df['cumulative'] = (1 + df['strategy_returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['cumulative'].iloc[-1])**(252/len(df)) - 1
    daily_returns = df['strategy_returns'].dropna()
    sharpe = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    downside_returns = daily_returns[daily_returns < 0]
    sortino = np.sqrt(252) * daily_returns.mean() / downside_returns.std()
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
    """Print the performance results."""
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve and compare with buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['cumulative'], label=f'{strategy_name} Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'Equity Curve for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Check today's signal based on the most recent data."""
    df = download_data(ticker, start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    last_signal = df['signal'].iloc[-1]
    position = "Long" if last_signal == 1 else "No Position"
    print(f"Today's Signal for {ticker}: {position}")

def main():
    parser = argparse.ArgumentParser(description='Time-Based Exit Strategy (4-Bar Exit)')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (default: 2010-01-01)')
    parser.add_argument('--end', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='End date (default: today)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve if set')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, "4-Bar Exit")

if __name__ == "__main__":
    main()
```