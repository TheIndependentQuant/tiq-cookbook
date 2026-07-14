```python
# 03 - ADX + Directional Filter
# Source: The Independent Quant | theindependentquant.com
# This strategy uses the Average Directional Index (ADX) to determine the strength of a trend in SPY and employs directional filters to decide trade direction. A buy signal is generated when the ADX is above a threshold and the +DI crosses above the -DI, indicating a strong upward trend. Conversely, a sell signal is triggered when the ADX is above the threshold and the -DI crosses above the +DI, signaling a strong downward trend. This approach aims to capitalize on sustained trends, allowing traders to profit from both bullish and bearish conditions.
# References:
# (No external references)
# Usage: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-10 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Constants
ADX_THRESHOLD = 25
LOOKBACK_PERIOD = 14

def download_data(ticker, start, end):
    """Download historical data for a given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the ADX + Directional Filter signals."""
    df['+DI'] = 100 * (df['High'].diff() > df['Low'].diff()).astype(int)
    df['-DI'] = 100 * (df['Low'].diff() > df['High'].diff()).astype(int)
    df['TR'] = df[['High', 'Low', 'Close']].max(axis=1) - df[['High', 'Low', 'Close']].min(axis=1)
    df['ATR'] = df['TR'].rolling(window=LOOKBACK_PERIOD).mean()
    df['+DI'] = df['+DI'].rolling(window=LOOKBACK_PERIOD).mean() / df['ATR'] * 100
    df['-DI'] = df['-DI'].rolling(window=LOOKBACK_PERIOD).mean() / df['ATR'] * 100
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].rolling(window=LOOKBACK_PERIOD).mean()

    df['signal'] = 0
    df.loc[(df['ADX'] > ADX_THRESHOLD) & (df['+DI'] > df['-DI']), 'signal'] = 1
    df.loc[(df['ADX'] > ADX_THRESHOLD) & (df['-DI'] > df['+DI']), 'signal'] = -1
    df['signal'] = df['signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy."""
    df['strategy_returns'] = df['signal'] * df['Close'].pct_change()
    df['cumulative'] = (1 + df['strategy_returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['cumulative'].iloc[-1])**(252/len(df)) - 1
    sharpe = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
    downside_std = df[df['strategy_returns'] < 0]['strategy_returns'].std()
    sortino = df['strategy_returns'].mean() / downside_std * np.sqrt(252)
    df['drawdown'] = df['cumulative'] / df['cumulative'].cummax() - 1
    max_drawdown = df['drawdown'].min()
    calmar = cagr / abs(max_drawdown)

    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print the performance metrics."""
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve versus buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['cumulative'], label=f'{strategy_name} Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Get today's signal for the given ticker."""
    df = download_data(ticker, '2023-01-01', pd.Timestamp.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    signal = df['signal'].iloc[-1]
    position = "Long" if signal == 1 else "Short" if signal == -1 else "Neutral"
    print(f"Today's signal for {ticker} is {position}.")

def main():
    parser = argparse.ArgumentParser(description='ADX + Directional Filter Strategy')
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
        plot_results(df, args.ticker, "ADX + Directional Filter")

if __name__ == "__main__":
    main()
```