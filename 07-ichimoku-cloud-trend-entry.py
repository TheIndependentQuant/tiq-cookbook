```python
#07 - Ichimoku Cloud Trend Entry
# Source: The Independent Quant | theindependentquant.com
# The Ichimoku Cloud Trend Entry strategy is a systematic trading approach designed to identify and capitalize on trends in the SPY, which is an ETF that tracks the S&P 500 index. This strategy uses the Ichimoku Cloud, a comprehensive technical analysis tool that provides insights into trend direction, momentum, and support/resistance levels. A buy signal is generated when the price of SPY crosses above the cloud, indicating a potential upward trend, while a sell signal occurs when the price falls below the cloud, suggesting a downward trend.
# References:
# (No external references)
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and plot flag.

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_SPAN_B_PERIOD = 52

def download_data(ticker, start, end):
    """Download historical data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute Ichimoku Cloud signals."""
    high_9 = df['High'].rolling(window=TENKAN_PERIOD).max()
    low_9 = df['Low'].rolling(window=TENKAN_PERIOD).min()
    df['Tenkan_sen'] = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=KIJUN_PERIOD).max()
    low_26 = df['Low'].rolling(window=KIJUN_PERIOD).min()
    df['Kijun_sen'] = (high_26 + low_26) / 2

    df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(KIJUN_PERIOD)

    high_52 = df['High'].rolling(window=SENKOU_SPAN_B_PERIOD).max()
    low_52 = df['Low'].rolling(window=SENKOU_SPAN_B_PERIOD).min()
    df['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(KIJUN_PERIOD)

    df['Signal'] = np.where(df['Close'] > df['Senkou_Span_A'], 1, 0)
    df['Signal'] = np.where(df['Close'] < df['Senkou_Span_B'], -1, df['Signal'])
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias

    return df

def backtest(df):
    """Backtest the strategy."""
    df['Strategy_Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1])**(252/len(df)) - 1
    daily_returns = df['Strategy_Returns'].dropna()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    downside_std = daily_returns[daily_returns < 0].std()
    sortino = daily_returns.mean() / downside_std * np.sqrt(252)
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
    """Print performance results."""
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot equity curve vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label='Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f"{strategy_name} on {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Print today's signal and position."""
    end = datetime.today().strftime('%Y-%m-%d')
    df = download_data(ticker, '2023-01-01', end)
    df = compute_signal(df)
    print(f"Today's signal for {ticker}: {df['Signal'].iloc[-1]}")

def main():
    parser = argparse.ArgumentParser(description='Ichimoku Cloud Trend Entry Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', default='2010-01-01', help='Start date (default: 2010-01-01)')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help='End date (default: today)')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve if set')

    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, '#07 - Ichimoku Cloud Trend Entry')

    todays_signal(args.ticker)

if __name__ == "__main__":
    main()
```