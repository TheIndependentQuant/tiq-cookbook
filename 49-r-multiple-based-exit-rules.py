```python
# 49 - R-Multiple-Based Exit Rules
# Source: The Independent Quant | theindependentquant.com
# The R-Multiple-Based Exit Rules strategy for SPY aims to optimize trade exits using predefined risk-reward ratios. It calculates an "R-multiple," the ratio of a trade's profit or loss to its initial risk, and exits trades when a specific R-multiple is achieved. This structured approach helps capture gains and control losses, particularly effective in trending markets.
# References:
# (No external references)
# Usage: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
INITIAL_RISK = 0.02  # 2% initial risk
TARGET_R_MULTIPLE = 2.0  # Exit at 2R
RISK_FREE_RATE = 0.01  # 1% risk-free rate

def download_data(ticker, start, end):
    """Download historical data for a given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute R-Multiple-Based Exit Rules signal."""
    df['Signal'] = 0
    df['Entry'] = df['Close']
    df['Stop'] = df['Entry'] * (1 - INITIAL_RISK)
    df['Target'] = df['Entry'] + (df['Entry'] - df['Stop']) * TARGET_R_MULTIPLE

    for i in range(1, len(df)):
        if df['Close'].iloc[i] >= df['Target'].iloc[i-1]:
            df['Signal'].iloc[i] = 1  # Exit signal

    df['Signal'] = df['Signal'].shift(1).fillna(0)
    return df

def backtest(df):
    """Backtest the strategy and calculate returns."""
    df['Position'] = df['Signal'].replace(to_replace=0, method='ffill')
    df['Strategy_Returns'] = df['Position'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    total_return = df['Cumulative'].iloc[-1] - 1
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1
    sharpe = (df['Strategy_Returns'].mean() - RISK_FREE_RATE / 252) / df['Strategy_Returns'].std() * np.sqrt(252)
    downside_std = df['Strategy_Returns'][df['Strategy_Returns'] < 0].std()
    sortino = (df['Strategy_Returns'].mean() - RISK_FREE_RATE / 252) / downside_std * np.sqrt(252)
    max_drawdown = (df['Cumulative'].cummax() - df['Cumulative']).max() / df['Cumulative'].cummax().max()
    calmar = cagr / max_drawdown

    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print the performance metrics."""
    print(f"Performance for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy & Hold')
    plt.title(f'{strategy_name} vs Buy & Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Print today's signal and position."""
    df = download_data(ticker, start=datetime.now().strftime('%Y-%m-%d'), end=datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    print(f"Today's signal for {ticker}: {df['Signal'].iloc[-1]}")

def main():
    """Main function to wire everything together."""
    parser = argparse.ArgumentParser(description='R-Multiple-Based Exit Rules Strategy')
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
        plot_results(df, args.ticker, "R-Multiple-Based Exit Rules")

if __name__ == "__main__":
    main()
```