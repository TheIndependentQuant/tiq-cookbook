```python
# 38 - Double Bottom with Breakout Confirmation
# Source: The Independent Quant | theindependentquant.com
# The Double Bottom with Breakout Confirmation strategy identifies a potential trend reversal by detecting a double bottom pattern in the price of SPY. This pattern occurs when the price hits a low, rebounds, drops again to a similar level, and then rises. The strategy requires a breakout above the resistance level formed between the two bottoms to confirm the pattern. This breakout suggests increased buying interest and potential for a bullish trend reversal. The strategy aims to capitalize on the potential price increase following the breakout.
# References:
# https://www.merriam-webster.com/dictionary/double
# Usage: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-10 --plot

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Constants
LOOKBACK_PERIOD = 20
DOUBLE_BOTTOM_THRESHOLD = 0.02

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the Double Bottom with Breakout Confirmation signal."""
    df['Low1'] = df['Low'].rolling(window=LOOKBACK_PERIOD).min()
    df['Low2'] = df['Low'].shift(LOOKBACK_PERIOD).rolling(window=LOOKBACK_PERIOD).min()
    df['Resistance'] = df['High'].shift(LOOKBACK_PERIOD).rolling(window=LOOKBACK_PERIOD).max()
    
    df['DoubleBottom'] = ((df['Low'] >= df['Low2'] * (1 - DOUBLE_BOTTOM_THRESHOLD)) &
                          (df['Low'] <= df['Low2'] * (1 + DOUBLE_BOTTOM_THRESHOLD)))
    
    df['Breakout'] = df['Close'] > df['Resistance']
    df['Signal'] = np.where(df['DoubleBottom'] & df['Breakout'], 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy and calculate returns."""
    df['StrategyReturns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['StrategyReturns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1] ** (1 / ((df.index[-1] - df.index[0]).days / 365.25))) - 1
    returns = df['StrategyReturns'].dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    downside_std = returns[returns < 0].std()
    sortino = returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else np.nan
    max_drawdown = ((df['Cumulative'].cummax() - df['Cumulative']).max() / df['Cumulative'].cummax()).max()
    calmar = cagr / max_drawdown if max_drawdown > 0 else np.nan
    
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
    print(f"CAGR: {perf['CAGR']:.2%}")
    print(f"Sharpe Ratio: {perf['Sharpe']:.2f}")
    print(f"Sortino Ratio: {perf['Sortino']:.2f}")
    print(f"Calmar Ratio: {perf['Calmar']:.2f}")
    print(f"Max Drawdown: {perf['Max Drawdown']:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve vs buy-and-hold."""
    df['BuyHold'] = (1 + df['Close'].pct_change()).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label=f"{strategy_name} Strategy")
    plt.plot(df['BuyHold'], label="Buy and Hold")
    plt.title(f"{strategy_name} Strategy vs Buy and Hold for {ticker}")
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Print today's signal based on recent data."""
    df = download_data(ticker, start='2023-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    if df['Signal'].iloc[-1] == 1:
        print(f"Today's signal for {ticker}: BUY")
    else:
        print(f"Today's signal for {ticker}: HOLD")

def main():
    parser = argparse.ArgumentParser(description="Double Bottom with Breakout Confirmation Strategy")
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', default='2010-01-01', help='Start date (default: 2010-01-01)')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help='End date (default: today)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve if set')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, "Double Bottom with Breakout Confirmation")

if __name__ == "__main__":
    main()
```