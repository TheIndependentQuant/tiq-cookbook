```python
# 25 - RSI + EMA Momentum Confirmation
# Source: The Independent Quant | theindependentquant.com
# The RSI + EMA Momentum Confirmation strategy is a systematic trading approach specifically designed for the SPY, which is the ETF tracking the S&P 500 index. This strategy combines two popular technical indicators: the Relative Strength Index (RSI) and the Exponential Moving Average (EMA). The RSI is used to measure the speed and change of price movements, providing insight into whether the SPY is overbought or oversold. An RSI value above 70 typically indicates overbought conditions, while a value below 30 suggests oversold conditions. The EMA, on the other hand, smooths out price data to identify the direction of the trend. By using a 50-day EMA, the strategy aims to confirm the momentum direction.
# References:
# (No external references)
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and a plot flag to visualize results.

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
RSI_PERIOD = 14
EMA_PERIOD = 50
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute RSI + EMA signals."""
    df['RSI'] = compute_rsi(df['Close'], RSI_PERIOD)
    df['EMA'] = df['Close'].ewm(span=EMA_PERIOD, adjust=False).mean()
    
    df['Signal'] = 0
    df.loc[(df['RSI'].shift(1) < RSI_OVERSOLD) & (df['RSI'] > RSI_OVERSOLD) & (df['Close'] > df['EMA']), 'Signal'] = 1
    df.loc[(df['RSI'].shift(1) > RSI_OVERBOUGHT) & (df['RSI'] < RSI_OVERBOUGHT) & (df['Close'] < df['EMA']), 'Signal'] = -1
    
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
    df['Position'] = df['Signal'].shift(1).fillna(0)
    df['Strategy_Returns'] = df['Position'] * df['Close'].pct_change()
    df['Cumulative'] = (1 + df['Strategy_Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    returns = df['Strategy_Returns'].dropna()
    cumulative_return = df['Cumulative'].iloc[-1] - 1
    cagr = (df['Cumulative'].iloc[-1]) ** (252 / len(df)) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    downside_std = returns[returns < 0].std()
    sortino = returns.mean() / downside_std * np.sqrt(252)
    max_drawdown = ((df['Cumulative'].cummax() - df['Cumulative']) / df['Cumulative'].cummax()).max()
    calmar = cagr / max_drawdown
    
    return {
        'CAGR': cagr,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Calmar Ratio': calmar,
        'Max Drawdown': max_drawdown,
        'Cumulative Return': cumulative_return
    }

def print_results(perf, ticker):
    """Print performance results."""
    print(f"Performance for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve and buy-and-hold comparison."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative'], label=f'{strategy_name} Strategy')
    plt.plot((1 + df['Close'].pct_change()).cumprod(), label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Get today's signal for the given ticker."""
    df = download_data(ticker, start='2022-01-01', end=datetime.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    print(f"Today's signal for {ticker}: {df['Signal'].iloc[-1]}")

def main():
    parser = argparse.ArgumentParser(description='RSI + EMA Momentum Confirmation Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.today().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve')
    
    args = parser.parse_args()
    
    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)
    
    if args.plot:
        plot_results(df, args.ticker, "RSI + EMA Momentum Confirmation")
    
    todays_signal(args.ticker)

if __name__ == '__main__':
    main()
```