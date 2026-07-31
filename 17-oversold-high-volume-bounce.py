```python
# 17 - Oversold + High Volume Bounce
# Source: The Independent Quant | theindependentquant.com
# 
# The Oversold + High Volume Bounce strategy identifies potential short-term reversals in the SPY ETF by focusing on two signals: an oversold condition and a high volume spike. An oversold condition is detected using the Relative Strength Index (RSI) with a value below 30. The high volume component is triggered when trading volume is significantly above average. The strategy predicts that these conditions indicate a likely bounce back in the short term, allowing traders to profit from the subsequent price recovery.
# 
# References:
# (No external references)
# 
# Usage instructions:
# Run the script with optional arguments for ticker, start date, end date, and plot flag. Example:
# python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
RSI_PERIOD = 14
RSI_OVERSOLD = 30
VOLUME_WINDOW = 20
VOLUME_MULTIPLIER = 1.5

def download_data(ticker, start, end):
    """Download historical data for a given ticker from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute trading signals based on RSI and volume."""
    df['RSI'] = compute_rsi(df['Adj Close'], RSI_PERIOD)
    df['AvgVolume'] = df['Volume'].rolling(window=VOLUME_WINDOW).mean()
    df['Signal'] = np.where((df['RSI'] < RSI_OVERSOLD) & 
                            (df['Volume'] > df['AvgVolume'] * VOLUME_MULTIPLIER), 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def compute_rsi(series, period):
    """Compute the Relative Strength Index (RSI) for a given series."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest(df):
    """Backtest the strategy and compute returns."""
    df['StrategyReturns'] = df['Signal'] * df['Adj Close'].pct_change()
    df['Cumulative'] = (1 + df['StrategyReturns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics for the strategy."""
    cagr = (df['Cumulative'].iloc[-1] ** (252/len(df))) - 1
    strategy_returns = df['StrategyReturns'].dropna()
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    downside_std = strategy_returns[strategy_returns < 0].std() * np.sqrt(252)
    sortino = strategy_returns.mean() / downside_std
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
    """Print the performance metrics in a formatted table."""
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve of the strategy vs buy-and-hold."""
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot(df.index, (1 + df['Adj Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Print today's trading signal based on the latest data."""
    df = download_data(ticker, start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    latest_signal = df['Signal'].iloc[-1]
    print(f"Today's signal for {ticker}: {'Buy' if latest_signal == 1 else 'Hold'}")

def main():
    parser = argparse.ArgumentParser(description='Oversold + High Volume Bounce Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, 'Oversold + High Volume Bounce')

    todays_signal(args.ticker)

if __name__ == '__main__':
    main()
```