```python
# 13 - CCI Pullback Long
# Source: The Independent Quant | theindependentquant.com
# 
# The CCI Pullback Long strategy is a systematic trading approach that focuses on identifying potential buying opportunities in the SPDR S&P 500 ETF Trust (SPY) by exploiting mean reversion tendencies in the market. This strategy uses the Commodity Channel Index (CCI), a momentum-based oscillator, to identify conditions where the SPY might be oversold and poised for a rebound. The strategy operates under the assumption that when the CCI drops below -100, the SPY has deviated significantly from its average price and is likely to revert back. By entering long positions when the CCI crosses above -100, the strategy predicts a short-term price recovery.
# 
# References:
# https://cci-online.org/
# 
# Usage instructions:
# Run this script from the command line with optional arguments for ticker, start date, end date, and plot flag.
# Example: python cci_pullback.py --ticker SPY --start 2010-01-01 --end 2023-10-10 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
CCI_PERIOD = 20
CCI_THRESHOLD = -100
RISK_FREE_RATE = 0.01

def download_data(ticker='SPY', start='2010-01-01', end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(CCI_PERIOD).mean()
    md = tp.rolling(CCI_PERIOD).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['CCI'] = (tp - ma) / (0.015 * md)
    df['Signal'] = np.where((df['CCI'].shift(1) < CCI_THRESHOLD) & (df['CCI'] > CCI_THRESHOLD), 1, 0)
    return df

def backtest(df):
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
    df['Cumulative_Strategy_Returns'] = (1 + df['Strategy_Returns']).cumprod()
    df['Cumulative_Market_Returns'] = (1 + df['Close'].pct_change()).cumprod()
    return df

def performance(df):
    cagr = (df['Cumulative_Strategy_Returns'].iloc[-1] ** (252.0/len(df))) - 1
    excess_returns = df['Strategy_Returns'] - RISK_FREE_RATE / 252
    sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    downside_returns = np.where(df['Strategy_Returns'] < 0, df['Strategy_Returns'], 0)
    sortino_ratio = np.sqrt(252) * excess_returns.mean() / np.std(downside_returns)
    max_drawdown = ((df['Cumulative_Strategy_Returns'].cummax() - df['Cumulative_Strategy_Returns']) / df['Cumulative_Strategy_Returns'].cummax()).max()
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
        print(f"{key}: {value:.2%}")

def plot_results(df, ticker, strategy_name):
    plt.figure(figsize=(14, 7))
    plt.plot(df['Cumulative_Strategy_Returns'], label=f'{strategy_name} Strategy')
    plt.plot(df['Cumulative_Market_Returns'], label='Buy and Hold')
    plt.title(f'{strategy_name} vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    df = download_data(ticker, start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    latest_signal = df['Signal'].iloc[-1]
    print(f"Today's signal for {ticker}: {'Buy' if latest_signal == 1 else 'No Action'}")

def main():
    parser = argparse.ArgumentParser(description='CCI Pullback Long Strategy')
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
        plot_results(df, args.ticker, 'CCI Pullback Long')

if __name__ == '__main__':
    main()
```