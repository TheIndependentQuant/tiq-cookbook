```python
# 04 - SuperTrend Entry/Exit
# Source: The Independent Quant | theindependentquant.com
# The SuperTrend Entry/Exit strategy is a trend-following approach designed to capture significant price movements in SPY, an ETF tracking the S&P 500. It uses the SuperTrend indicator, based on ATR and a multiplier, to signal entry and exit points. A price crossing above the SuperTrend line signals a buy, while crossing below signals a sell. The strategy aims to capture trends by focusing on substantial price movements, filtering out minor fluctuations.
# References:
# (No external references)
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and plot flag. Example: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import pandas as pd
import numpy as np
import yfinance as yf
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
ATR_PERIOD = 10
ATR_MULTIPLIER = 3.0

def download_data(ticker, start, end):
    """Download historical data for a given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the SuperTrend signal."""
    df['ATR'] = df['High'].rolling(window=ATR_PERIOD).max() - df['Low'].rolling(window=ATR_PERIOD).min()
    df['ATR'] = df['ATR'].rolling(window=ATR_PERIOD).mean()
    df['UpperBand'] = ((df['High'] + df['Low']) / 2) + (ATR_MULTIPLIER * df['ATR'])
    df['LowerBand'] = ((df['High'] + df['Low']) / 2) - (ATR_MULTIPLIER * df['ATR'])
    
    df['SuperTrend'] = np.nan
    df['Signal'] = 0
    
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['UpperBand'].iloc[i-1]:
            df['SuperTrend'].iloc[i] = df['LowerBand'].iloc[i]
            df['Signal'].iloc[i] = 1
        elif df['Close'].iloc[i] < df['LowerBand'].iloc[i-1]:
            df['SuperTrend'].iloc[i] = df['UpperBand'].iloc[i]
            df['Signal'].iloc[i] = -1
        else:
            df['SuperTrend'].iloc[i] = df['SuperTrend'].iloc[i-1]
            df['Signal'].iloc[i] = df['Signal'].iloc[i-1]
    
    df['Position'] = df['Signal'].shift(1)
    return df

def backtest(df):
    """Backtest the strategy."""
    df['Strategy_Returns'] = df['Position'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1])**(252/len(df)) - 1
    returns = df['Strategy_Returns'].mean() * 252
    volatility = df['Strategy_Returns'].std() * np.sqrt(252)
    sharpe = returns / volatility
    downside_std = df[df['Strategy_Returns'] < 0]['Strategy_Returns'].std() * np.sqrt(252)
    sortino = returns / downside_std
    max_drawdown = (df['Cumulative'].cummax() - df['Cumulative']).max()
    calmar = returns / max_drawdown

    return {
        'CAGR': cagr,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Calmar': calmar,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print performance results."""
    print(f"Performance for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} vs Buy and Hold for {ticker}')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Get today's signal for the given ticker."""
    end = datetime.now().strftime('%Y-%m-%d')
    df = download_data(ticker, start='2023-01-01', end=end)
    df = compute_signal(df)
    print(f"Today's signal for {ticker}: {df['Signal'].iloc[-1]}")
    print(f"Current position: {df['Position'].iloc[-1]}")

def main():
    parser = argparse.ArgumentParser(description='SuperTrend Entry/Exit Strategy')
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
        plot_results(df, args.ticker, "SuperTrend Entry/Exit")

if __name__ == "__main__":
    main()
```