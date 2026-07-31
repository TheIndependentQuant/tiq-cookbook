```python
# 20 - Gap Down Reversion with Close Filter
# Source: The Independent Quant | theindependentquant.com
# This strategy identifies potential mean reversion opportunities in the SPY ETF by looking for instances where the SPY opens significantly lower than its previous close (a gap down) and then closes higher than its open. This setup suggests a potential short-term reversal, predicting a price increase in the following days. The strategy exploits overreactions in the market, aiming to capitalize on short-term price inefficiencies.
# References:
# https://www.gap.com/
# Usage: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Constants
LOOKBACK_WINDOW = 1  # Lookback period for gap down
CLOSE_FILTER = True  # Use close filter

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the trading signal for the Gap Down Reversion with Close Filter strategy."""
    df['Prev_Close'] = df['Close'].shift(1)
    df['Gap_Down'] = df['Open'] < df['Prev_Close']
    df['Close_Higher'] = df['Close'] > df['Open']
    df['Signal'] = np.where(df['Gap_Down'] & df['Close_Higher'], 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy and compute returns."""
    df['Strategy_Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    total_return = df['Cumulative'].iloc[-1] - 1
    cagr = (df['Cumulative'].iloc[-1]) ** (252 / len(df)) - 1
    sharpe = df['Strategy_Returns'].mean() / df['Strategy_Returns'].std() * np.sqrt(252)
    sortino = df['Strategy_Returns'].mean() / df[df['Strategy_Returns'] < 0]['Strategy_Returns'].std() * np.sqrt(252)
    max_drawdown = ((df['Cumulative'].cummax() - df['Cumulative']) / df['Cumulative'].cummax()).max()
    calmar = cagr / max_drawdown
    return {
        'Total Return': total_return,
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_drawdown,
        'Calmar Ratio': calmar
    }

def print_results(perf, ticker):
    """Print the performance metrics."""
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve of the strategy."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} on {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print today's signal."""
    df = download_data(ticker, start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    latest_signal = df['Signal'].iloc[-1]
    print(f"Today's Signal for {ticker}: {'Buy' if latest_signal == 1 else 'No Action'}")

def main():
    """Main function to wire everything together."""
    parser = argparse.ArgumentParser(description='Gap Down Reversion with Close Filter Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol for the stock')
    parser.add_argument('--start', default='2010-01-01', help='Start date for historical data')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help='End date for historical data')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve if set')
    
    args = parser.parse_args()
    
    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)
    
    if args.plot:
        plot_results(df, args.ticker, 'Gap Down Reversion with Close Filter')
    
    todays_signal(args.ticker)

if __name__ == "__main__":
    main()
```