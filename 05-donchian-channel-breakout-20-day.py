```python
# 05 - Donchian Channel Breakout (20-day)
# Source: The Independent Quant | theindependentquant.com
# The Donchian Channel Breakout (20-day) strategy is a trend-following approach designed to capture momentum in the SPY, which is an ETF that tracks the S&P 500 index. This strategy identifies potential breakout points by calculating the highest high and the lowest low over the past 20 trading days. The upper band of the Donchian Channel represents the 20-day high, while the lower band represents the 20-day low. When the price of SPY closes above the upper band, it signals a potential upward breakout, indicating a buy signal. Conversely, a close below the lower band suggests a downward breakout, indicating a sell signal.
# References:
# https://jane.app/sign-in
# Usage: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Constants
LOOKBACK_PERIOD = 20

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the Donchian Channel Breakout signals."""
    df['20D_High'] = df['High'].rolling(window=LOOKBACK_PERIOD).max()
    df['20D_Low'] = df['Low'].rolling(window=LOOKBACK_PERIOD).min()
    df['Signal'] = 0
    df.loc[df['Close'] > df['20D_High'].shift(1), 'Signal'] = 1
    df.loc[df['Close'] < df['20D_Low'].shift(1), 'Signal'] = -1
    return df

def backtest(df):
    """Backtest the strategy and compute returns."""
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1]) ** (252.0 / len(df)) - 1
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
    """Get today's signal for the given ticker."""
    df = download_data(ticker, '2023-09-01', pd.Timestamp.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    last_signal = df['Signal'].iloc[-1]
    position = "Long" if last_signal == 1 else "Short" if last_signal == -1 else "Neutral"
    print(f"Today's signal for {ticker}: {position}")

def main():
    parser = argparse.ArgumentParser(description='Donchian Channel Breakout Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=pd.Timestamp.today().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve')

    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, "Donchian Channel Breakout (20-day)")

    todays_signal(args.ticker)

if __name__ == "__main__":
    main()
```