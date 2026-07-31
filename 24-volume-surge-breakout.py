```python
# 24 - Volume Surge + Breakout
# Source: The Independent Quant | theindependentquant.com
# The Volume Surge + Breakout strategy is a systematic trading approach designed to capitalize on momentum in the SPDR S&P 500 ETF Trust (SPY). This strategy identifies potential breakout points by analyzing volume surges in conjunction with price movements. Specifically, it looks for instances where the trading volume of SPY significantly exceeds its average over a specified period, indicating heightened investor interest. When this volume surge coincides with a breakout above a defined resistance level, it generates a buy signal. The underlying assumption is that such conditions suggest a strong likelihood of continued upward price movement, as increased volume often reflects a shift in market sentiment or the entry of large institutional players.
# References:
# https://www.omnicalculator.com/math/volume
# Usage: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-31 --plot

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Constants
VOLUME_LOOKBACK = 20
BREAKOUT_LOOKBACK = 50
VOLUME_THRESHOLD = 1.5

def download_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

def compute_signal(df):
    df['AvgVolume'] = df['Volume'].rolling(window=VOLUME_LOOKBACK).mean()
    df['VolumeSurge'] = df['Volume'] > VOLUME_THRESHOLD * df['AvgVolume']
    df['HighBreakout'] = df['Close'] > df['Close'].rolling(window=BREAKOUT_LOOKBACK).max().shift(1)
    df['Signal'] = np.where(df['VolumeSurge'] & df['HighBreakout'], 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    df['StrategyReturns'] = df['Signal'] * df['Close'].pct_change()
    df['CumulativeStrategyReturns'] = (1 + df['StrategyReturns']).cumprod()
    df['CumulativeMarketReturns'] = (1 + df['Close'].pct_change()).cumprod()
    return df

def performance(df):
    total_days = (df.index[-1] - df.index[0]).days
    cagr = (df['CumulativeStrategyReturns'].iloc[-1]) ** (365.0 / total_days) - 1
    daily_returns = df['StrategyReturns'].dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    downside_std = daily_returns[daily_returns < 0].std()
    sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_std
    max_drawdown = (df['CumulativeStrategyReturns'].cummax() - df['CumulativeStrategyReturns']).max()
    calmar_ratio = cagr / max_drawdown
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    print(f"Performance for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    plt.figure(figsize=(14, 7))
    plt.plot(df['CumulativeStrategyReturns'], label=f'{strategy_name} Strategy')
    plt.plot(df['CumulativeMarketReturns'], label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold on {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    df = download_data(ticker, datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    signal = df['Signal'].iloc[-1]
    position = 'Long' if signal == 1 else 'Neutral'
    print(f"Today's signal for {ticker}: {position}")

def main():
    parser = argparse.ArgumentParser(description='Volume Surge + Breakout Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol')
    parser.add_argument('--start', default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, "Volume Surge + Breakout")

if __name__ == "__main__":
    main()
```