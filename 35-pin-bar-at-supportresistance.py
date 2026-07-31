```python
# 35 - Pin Bar at Support/Resistance
# Source: The Independent Quant | theindependentquant.com
# The Pin Bar at Support/Resistance strategy is a price action-based approach designed for trading the SPDR S&P 500 ETF Trust (SPY). It identifies pin bars, which are candlestick patterns with a small body and long wick, at key support or resistance levels. A bullish pin bar at support suggests a potential upward reversal, while a bearish pin bar at resistance suggests a potential downward reversal. This strategy aims to capitalize on these reversals, leveraging the psychological and behavioral aspects of market participants at these critical levels.
# References:
# (No external references)
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and plot flag to see the strategy's performance.

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Constants
LOOKBACK_PERIOD = 20
PIN_BAR_THRESHOLD = 0.6

def download_data(ticker='SPY', start='2010-01-01', end=None):
    """Download historical data for the given ticker."""
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the trading signal based on Pin Bar at Support/Resistance."""
    df['Signal'] = 0
    df['Support'] = df['Low'].rolling(LOOKBACK_PERIOD).min()
    df['Resistance'] = df['High'].rolling(LOOKBACK_PERIOD).max()
    
    for i in range(LOOKBACK_PERIOD, len(df)):
        body_size = abs(df['Open'][i] - df['Close'][i])
        upper_wick = df['High'][i] - max(df['Open'][i], df['Close'][i])
        lower_wick = min(df['Open'][i], df['Close'][i]) - df['Low'][i]
        
        if lower_wick > body_size * PIN_BAR_THRESHOLD and df['Low'][i] <= df['Support'][i-1]:
            df.at[df.index[i], 'Signal'] = 1  # Bullish Pin Bar at Support
        elif upper_wick > body_size * PIN_BAR_THRESHOLD and df['High'][i] >= df['Resistance'][i-1]:
            df.at[df.index[i], 'Signal'] = -1  # Bearish Pin Bar at Resistance
    
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy and calculate returns."""
    df['Strategy_Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1])**(252/len(df)) - 1
    daily_returns = df['Strategy_Returns'].dropna()
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    sortino = daily_returns.mean() / daily_returns[daily_returns < 0].std() * np.sqrt(252)
    max_drawdown = (df['Cumulative'].cummax() - df['Cumulative']).max()
    calmar = cagr / max_drawdown if max_drawdown != 0 else np.nan
    
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
    """Plot the equity curve of the strategy vs. buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print the current signal."""
    end = datetime.today().strftime('%Y-%m-%d')
    start = (datetime.today() - pd.DateOffset(days=LOOKBACK_PERIOD + 10)).strftime('%Y-%m-%d')
    df = download_data(ticker, start=start, end=end)
    df = compute_signal(df)
    latest_signal = df['Signal'].iloc[-1]
    position = 'Long' if latest_signal == 1 else 'Short' if latest_signal == -1 else 'Neutral'
    print(f"Today's signal for {ticker}: {position}")

def main():
    """Main function to wire everything together."""
    parser = argparse.ArgumentParser(description='Pin Bar at Support/Resistance Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (default: 2010-01-01)')
    parser.add_argument('--end', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='End date (default: today)')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve if set')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)
    
    if args.plot:
        plot_results(df, args.ticker, 'Pin Bar at Support/Resistance')

    todays_signal(args.ticker)

if __name__ == '__main__':
    main()
```