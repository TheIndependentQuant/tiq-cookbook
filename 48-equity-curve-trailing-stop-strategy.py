```python
# 48 - Equity Curve Trailing Stop Strategy
# Source: The Independent Quant | theindependentquant.com
# This strategy manages risk and optimizes returns by applying a trailing stop mechanism based on the equity curve of the trading account, rather than the price of SPY itself. It adjusts the trailing stop upwards when the equity curve reaches a new high, securing gains. If the equity curve declines by a certain percentage from its peak, it triggers an exit, predicting potential market downturns. This approach helps capture gains during upward trends while minimizing losses during downturns.

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
TRAILING_STOP_PERCENT = 0.05  # 5% trailing stop
LOOKBACK_PERIOD = 252  # 1-year lookback for equity curve

def download_data(ticker, start, end):
    """Download historical data for a given ticker."""
    df = yf.download(ticker, start=start, end=end)
    df['Returns'] = df['Adj Close'].pct_change()
    return df

def compute_signal(df):
    """Compute the trading signal based on the equity curve trailing stop."""
    df['Equity'] = (1 + df['Returns']).cumprod()
    df['Equity Peak'] = df['Equity'].cummax()
    df['Drawdown'] = df['Equity'] / df['Equity Peak'] - 1
    df['Signal'] = np.where(df['Drawdown'] <= -TRAILING_STOP_PERCENT, 0, 1)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy and compute strategy returns."""
    df['Strategy Returns'] = df['Returns'] * df['Signal']
    df['Cumulative Strategy'] = (1 + df['Strategy Returns']).cumprod()
    df['Cumulative Buy & Hold'] = (1 + df['Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics for the strategy."""
    cagr = (df['Cumulative Strategy'].iloc[-1]) ** (252 / len(df)) - 1
    daily_return = df['Strategy Returns'].mean()
    daily_vol = df['Strategy Returns'].std()
    sharpe_ratio = daily_return / daily_vol * np.sqrt(252)
    downside_vol = df[df['Strategy Returns'] < 0]['Strategy Returns'].std()
    sortino_ratio = daily_return / downside_vol * np.sqrt(252)
    max_drawdown = df['Drawdown'].min()
    calmar_ratio = cagr / abs(max_drawdown)
    
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print the performance results in a formatted table."""
    print(f"Performance for {ticker}:")
    print(f"CAGR: {perf['CAGR']:.2%}")
    print(f"Sharpe Ratio: {perf['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio: {perf['Sortino Ratio']:.2f}")
    print(f"Calmar Ratio: {perf['Calmar Ratio']:.2f}")
    print(f"Max Drawdown: {perf['Max Drawdown']:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve of the strategy vs buy-and-hold."""
    plt.figure(figsize=(14, 7))
    plt.plot(df['Cumulative Strategy'], label='Strategy')
    plt.plot(df['Cumulative Buy & Hold'], label='Buy & Hold', linestyle='--')
    plt.title(f"{strategy_name} on {ticker}")
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Print today's trading signal based on the latest data."""
    df = download_data(ticker, '2023-01-01', datetime.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    last_signal = df['Signal'].iloc[-1]
    print(f"Today's signal for {ticker}: {'Hold' if last_signal == 1 else 'Exit'}")

def main():
    """Main function to run the strategy."""
    parser = argparse.ArgumentParser(description='Equity Curve Trailing Stop Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', default='2010-01-01', help='Start date (default: 2010-01-01)')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help='End date (default: today)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve')
    
    args = parser.parse_args()
    
    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    
    print_results(perf, args.ticker)
    
    if args.plot:
        plot_results(df, args.ticker, "Equity Curve Trailing Stop Strategy")
    
    todays_signal(args.ticker)

if __name__ == "__main__":
    main()
```