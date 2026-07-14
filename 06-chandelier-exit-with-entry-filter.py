```python
# 06 - Chandelier Exit with Entry Filter
# Source: The Independent Quant | theindependentquant.com
# 
# The Chandelier Exit with Entry Filter strategy is a trend-following approach specifically designed for trading the SPY, an ETF that tracks the S&P 500 index. The strategy employs the Chandelier Exit, a volatility-based indicator, to signal potential exit points by trailing the stop-loss level above or below the current price. This is calculated using the highest high or lowest low over a specific period, adjusted by a multiple of the Average True Range (ATR). The entry filter, often a momentum indicator like the Relative Strength Index (RSI), is used to determine optimal entry points by identifying overbought or oversold conditions in the market.
# 
# References:
# (No external references)
# 
# Usage instructions:
# Run the script with optional arguments for ticker, start date, end date, and plot flag. Example:
# python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
ATR_PERIOD = 22
ATR_MULTIPLIER = 3.0
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute Chandelier Exit with Entry Filter signals."""
    df['ATR'] = df['High'].rolling(window=ATR_PERIOD).max() - df['Low'].rolling(window=ATR_PERIOD).min()
    df['ATR'] = df['ATR'].rolling(window=ATR_PERIOD).mean()
    
    df['Chandelier_Exit_Long'] = df['High'].rolling(window=ATR_PERIOD).max() - (df['ATR'] * ATR_MULTIPLIER)
    df['Chandelier_Exit_Short'] = df['Low'].rolling(window=ATR_PERIOD).min() + (df['ATR'] * ATR_MULTIPLIER)
    
    df['RSI'] = compute_rsi(df['Close'], RSI_PERIOD)
    
    df['Signal'] = 0
    df.loc[(df['Close'] > df['Chandelier_Exit_Long']) & (df['RSI'] < RSI_OVERSOLD), 'Signal'] = 1
    df.loc[(df['Close'] < df['Chandelier_Exit_Short']) & (df['RSI'] > RSI_OVERBOUGHT), 'Signal'] = -1
    
    df['Signal'] = df['Signal'].shift(1)
    return df

def compute_rsi(series, period):
    """Compute the Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def backtest(df):
    """Backtest the strategy."""
    df['Strategy_Returns'] = df['Signal'].shift(1) * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    df['Buy_and_Hold'] = (1 + df['Close'].pct_change()).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1]) ** (252 / len(df)) - 1
    sharpe = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252)
    sortino = (df['Strategy_Returns'].mean() / df['Strategy_Returns'][df['Strategy_Returns'] < 0].std()) * np.sqrt(252)
    max_drawdown = (df['Cumulative'].cummax() - df['Cumulative']).max()
    calmar = cagr / max_drawdown
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown': max_drawdown
    }

def print_results(perf, ticker):
    """Print formatted performance table."""
    print(f"Performance for {ticker}:")
    print(f"CAGR: {perf['CAGR']:.2%}")
    print(f"Sharpe Ratio: {perf['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio: {perf['Sortino Ratio']:.2f}")
    print(f"Calmar Ratio: {perf['Calmar Ratio']:.2f}")
    print(f"Max Drawdown: {perf['Max Drawdown']:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot equity curve vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot(df['Buy_and_Hold'], label='Buy and Hold')
    plt.title(f'{strategy_name} vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print current signal and position."""
    today = datetime.today().strftime('%Y-%m-%d')
    df = download_data(ticker, '2023-01-01', today)
    df = compute_signal(df)
    print(f"Today's signal for {ticker}: {df['Signal'].iloc[-1]}")

def main():
    """Main function to wire everything together via argparse."""
    parser = argparse.ArgumentParser(description='Chandelier Exit with Entry Filter Strategy')
    parser.add_argument('--ticker', default='SPY', help='Ticker symbol (default: SPY)')
    parser.add_argument('--start', default='2010-01-01', help='Start date (default: 2010-01-01)')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help='End date (default: today)')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve chart if set')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, "Chandelier Exit with Entry Filter")

if __name__ == '__main__':
    main()
```