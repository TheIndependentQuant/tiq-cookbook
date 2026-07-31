```python
#42 - Volatility Filter (ATR Rising = No Entry)
# Source: The Independent Quant | theindependentquant.com
# The Volatility Filter (ATR Rising = No Entry) strategy is designed to manage market entry points based on volatility conditions, specifically targeting the SPY ETF, which tracks the S&P 500 index. The strategy uses the Average True Range (ATR) as a volatility measure. If the ATR is rising, indicating increasing volatility, the strategy refrains from entering new positions. Conversely, if the ATR is stable or declining, the strategy allows for potential entry into the market. This approach aims to reduce exposure to sudden adverse price movements during periods of high volatility.
# References:
# https://www.investopedia.com/terms/v/volatility.asp
# Usage: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Constants
ATR_PERIOD = 14

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute the trading signal based on ATR rising condition."""
    df['ATR'] = df['High'].rolling(window=ATR_PERIOD).max() - df['Low'].rolling(window=ATR_PERIOD).min()
    df['ATR_diff'] = df['ATR'].diff()
    df['Signal'] = np.where(df['ATR_diff'] > 0, 0, 1)  # 0 = No Entry, 1 = Entry Allowed
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy and calculate returns."""
    df['Market_Return'] = df['Adj Close'].pct_change()
    df['Strategy_Return'] = df['Market_Return'] * df['Signal']
    df['Cumulative_Market'] = (1 + df['Market_Return']).cumprod()
    df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative_Strategy'].iloc[-1]) ** (252.0 / len(df)) - 1
    sharpe = np.mean(df['Strategy_Return']) / np.std(df['Strategy_Return']) * np.sqrt(252)
    sortino = np.mean(df['Strategy_Return']) / np.std(df[df['Strategy_Return'] < 0]['Strategy_Return']) * np.sqrt(252)
    max_drawdown = (df['Cumulative_Strategy'].cummax() - df['Cumulative_Strategy']).max()
    calmar = cagr / max_drawdown
    return {'CAGR': cagr, 'Sharpe': sharpe, 'Sortino': sortino, 'Calmar': calmar, 'Max Drawdown': max_drawdown}

def print_results(perf, ticker):
    """Print the performance results."""
    print(f"Performance Metrics for {ticker}:")
    for key, value in perf.items():
        print(f"{key}: {value:.2f}")

def plot_results(df, ticker, strategy_name):
    """Plot the equity curve of the strategy versus buy-and-hold."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['Cumulative_Market'], label='Buy and Hold')
    plt.plot(df['Cumulative_Strategy'], label=strategy_name)
    plt.title(f'{strategy_name} vs Buy and Hold for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Print today's signal based on recent data."""
    df = download_data(ticker, '2023-01-01', pd.Timestamp.today().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    print(f"Today's Signal for {ticker}: {'Entry Allowed' if df['Signal'].iloc[-1] == 1 else 'No Entry'}")

def main():
    """Main function to wire everything together."""
    parser = argparse.ArgumentParser(description='Volatility Filter (ATR Rising = No Entry) Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=pd.Timestamp.today().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot equity curve if set')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, '#42 - Volatility Filter (ATR Rising = No Entry)')

if __name__ == "__main__":
    main()
```