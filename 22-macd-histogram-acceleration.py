```python
# 22 - MACD Histogram Acceleration
# Source: The Independent Quant | theindependentquant.com
# The MACD Histogram Acceleration strategy is a momentum-based trading approach applied to SPY. It focuses on the acceleration of the MACD histogram, which is the rate of change of the histogram values over time. A positive acceleration suggests increasing bullish momentum, prompting a buy signal, while a negative acceleration indicates growing bearish momentum, leading to a sell signal. This strategy aims to profit from short- to medium-term price movements in SPY by capturing momentum shifts.
# References:
# (No external references)
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and plot flag.

import pandas as pd
import numpy as np
import yfinance as yf
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
SHORT_EMA = 12
LONG_EMA = 26
SIGNAL_SMOOTH = 9

def download_data(ticker='SPY', start='2010-01-01', end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    df['EMA12'] = df['Close'].ewm(span=SHORT_EMA, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=LONG_EMA, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=SIGNAL_SMOOTH, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    df['MACD_Hist_Accel'] = df['MACD_Hist'].diff()
    df['Signal'] = np.where(df['MACD_Hist_Accel'] > 0, 1, -1)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    df['Strategy_Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    df['Cumulative_Market'] = (1 + df['Close'].pct_change()).cumprod()
    return df

def performance(df):
    total_days = (df.index[-1] - df.index[0]).days
    cagr = (df['Cumulative_Strategy'].iloc[-1])**(365.0/total_days) - 1
    daily_returns = df['Strategy_Returns'].dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    downside_returns = daily_returns[daily_returns < 0]
    sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_returns.std()
    max_drawdown = ((df['Cumulative_Strategy'].cummax() - df['Cumulative_Strategy']).max() / df['Cumulative_Strategy'].cummax()).max()
    calmar_ratio = cagr / max_drawdown
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Cumulative_Strategy'], label='Strategy')
    plt.plot(df['Cumulative_Market'], label='Buy and Hold')
    plt.title(f"{strategy_name} on {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    df = download_data(ticker, start='2022-01-01')
    df = compute_signal(df)
    last_signal = df['Signal'].iloc[-1]
    position = 'Long' if last_signal == 1 else 'Short'
    print(f"Today's signal for {ticker}: {position}")

def main():
    parser = argparse.ArgumentParser(description='MACD Histogram Acceleration Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, "#22 - MACD Histogram Acceleration")

    todays_signal(args.ticker)

if __name__ == "__main__":
    main()
```