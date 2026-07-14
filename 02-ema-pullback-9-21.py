```python
# 02 - EMA Pullback (9-21)
# Source: The Independent Quant | theindependentquant.com
# 
# The EMA Pullback (9-21) strategy is a trend-following approach designed to capitalize on short-term price movements in the SPDR S&P 500 ETF Trust (SPY). This strategy uses two Exponential Moving Averages (EMAs) to identify potential entry points. Specifically, it calculates the 9-day EMA and the 21-day EMA of SPY's closing prices. The strategy generates a buy signal when the price of SPY pulls back to the 21-day EMA while the 9-day EMA is above the 21-day EMA. This indicates a potential continuation of an upward trend after a brief retracement.
#
# References:
# (No external references)
#
# Usage instructions:
# Run the script with optional arguments for ticker, start date, end date, and plot flag. For example:
# python ema_pullback.py --ticker SPY --start 2010-01-01 --end 2023-10-10 --plot

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf

# Constants
EMA_SHORT_PERIOD = 9
EMA_LONG_PERIOD = 21

def download_data(ticker, start, end):
    """Download historical data for the given ticker from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the EMA Pullback (9-21) signal."""
    df['EMA9'] = df['Close'].ewm(span=EMA_SHORT_PERIOD, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=EMA_LONG_PERIOD, adjust=False).mean()
    df['Signal'] = np.where((df['Close'] <= df['EMA21']) & (df['EMA9'] > df['EMA21']), 1, 0)
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy."""
    df['Strategy_Returns'] = df['Signal'] * df['Close'].pct_change()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns']).cumprod()
    df['Cumulative_Buy_and_Hold'] = (1 + df['Close'].pct_change()).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative_Strategy'].iloc[-1])**(252/len(df)) - 1
    excess_return = df['Strategy_Returns'].mean() * 252
    excess_volatility = df['Strategy_Returns'].std() * np.sqrt(252)
    sharpe_ratio = excess_return / excess_volatility
    downside_volatility = df[df['Strategy_Returns'] < 0]['Strategy_Returns'].std() * np.sqrt(252)
    sortino_ratio = excess_return / downside_volatility
    max_drawdown = (df['Cumulative_Strategy'].cummax() - df['Cumulative_Strategy']).max()
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
        print(f"{key}: {value:.2%}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve of the strategy vs buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative_Strategy'], label=f'{strategy_name} Strategy')
    plt.plot(df['Cumulative_Buy_and_Hold'], label='Buy and Hold')
    plt.title(f'{strategy_name} Strategy vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Print the current signal and position."""
    end_date = datetime.today().strftime('%Y-%m-%d')
    df = download_data(ticker, '2023-01-01', end_date)
    df = compute_signal(df)
    current_signal = df['Signal'].iloc[-1]
    print(f"Today's signal for {ticker}: {'Buy' if current_signal == 1 else 'Hold/Sell'}")

def main():
    parser = argparse.ArgumentParser(description='EMA Pullback (9-21) Strategy')
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
        plot_results(df, args.ticker, 'EMA Pullback (9-21)')
    
    todays_signal(args.ticker)

if __name__ == "__main__":
    main()
```