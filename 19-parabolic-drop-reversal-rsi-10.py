```python
# 19 - Parabolic Drop Reversal (RSI < 10)
# Source: The Independent Quant | theindependentquant.com
# The Parabolic Drop Reversal strategy focuses on identifying extreme oversold conditions in SPY, the ETF tracking the S&P 500 index, using the Relative Strength Index (RSI). Specifically, it targets instances when the RSI falls below 10, indicating a potential for a sharp reversal. By entering long positions at these points, the strategy aims to profit from the anticipated recovery in SPY's price. This approach leverages mean reversion, exploiting market overreactions that often lead to temporary bounces.

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import pandas_datareader.data as web

# Constants
RSI_THRESHOLD = 10
RSI_PERIOD = 14

def download_data(ticker, start, end):
    """Download historical data for the specified ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the trading signals based on RSI < 10."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    df['Signal'] = np.where(df['RSI'] < RSI_THRESHOLD, 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy."""
    df['Strategy_Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1] ** (252 / len(df))) - 1
    daily_return = df['Strategy_Returns'].mean()
    daily_vol = df['Strategy_Returns'].std()
    sharpe_ratio = daily_return / daily_vol * np.sqrt(252)
    downside_vol = df.loc[df['Strategy_Returns'] < 0, 'Strategy_Returns'].std()
    sortino_ratio = daily_return / downside_vol * np.sqrt(252)
    max_drawdown = (df['Cumulative'].cummax() - df['Cumulative']).max()
    calmar_ratio = cagr / max_drawdown
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe_ratio,
        'Sortino Ratio': sortino_ratio,
        'Calmar Ratio': calmar_ratio,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print the performance results."""
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve of the strategy."""
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Cumulative'], label=strategy_name)
    plt.plot(df.index, (1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Print today's signal and position."""
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - pd.Timedelta(days=RSI_PERIOD * 2)).strftime('%Y-%m-%d')
    df = download_data(ticker, start, end)
    df = compute_signal(df)
    signal = df['Signal'].iloc[-1]
    position = "Long" if signal == 1 else "No Position"
    print(f"Today's Signal for {ticker}: {signal} ({position})")

def main():
    parser = argparse.ArgumentParser(description='Parabolic Drop Reversal (RSI < 10) Strategy')
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
        plot_results(df, args.ticker, 'Parabolic Drop Reversal (RSI < 10)')

    todays_signal(args.ticker)

if __name__ == "__main__":
    main()
```