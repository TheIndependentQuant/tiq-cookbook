```python
# 50 - Fixed Fractional Position Sizing with Drawdown Cap
# Source: The Independent Quant | theindependentquant.com
# This strategy uses the RSI indicator to generate buy and sell signals for SPY, aiming to capitalize on mean-reverting price movements. It employs fixed fractional position sizing to manage risk and includes a drawdown cap to limit potential losses. Buy signals occur when RSI crosses above 30, and sell signals occur when RSI crosses below 70.
# References:
# (No external references)
# Usage instructions: Run the script with optional arguments for ticker, start date, end date, and plot flag. Example: python script.py --ticker SPY --start 2010-01-01 --end 2023-10-01 --plot

import argparse
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Constants
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
FRACTIONAL_SIZE = 0.1
DRAWDOWN_CAP = 0.2

def download_data(ticker, start, end):
    """Download historical data for the given ticker."""
    df = yf.download(ticker, start=start, end=end)
    return df

def compute_signal(df):
    """Compute buy/sell signals based on RSI."""
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Signal'] = 0
    df.loc[df['RSI'] < RSI_OVERSOLD, 'Signal'] = 1
    df.loc[df['RSI'] > RSI_OVERBOUGHT, 'Signal'] = -1
    df['Signal'] = df['Signal'].shift(1)  # Prevent look-ahead bias
    return df

def backtest(df):
    """Backtest the strategy with fixed fractional position sizing and drawdown cap."""
    df['Position'] = 0
    capital = 1.0
    peak = capital
    for i in range(1, len(df)):
        if df['Signal'].iloc[i] == 1:
            df['Position'].iloc[i] = FRACTIONAL_SIZE * capital / df['Adj Close'].iloc[i]
        elif df['Signal'].iloc[i] == -1:
            df['Position'].iloc[i] = 0
        capital = df['Position'].iloc[i] * df['Adj Close'].iloc[i]
        df['Strategy Value'] = capital
        peak = max(peak, capital)
        if (peak - capital) / peak > DRAWDOWN_CAP:
            df['Position'].iloc[i] = 0
            capital = 1.0
    df['Strategy Returns'] = df['Strategy Value'].pct_change().fillna(0)
    df['Cumulative'] = (1 + df['Strategy Returns']).cumprod()
    return df

def performance(df):
    """Calculate performance metrics."""
    cagr = (df['Cumulative'].iloc[-1]) ** (1 / ((df.index[-1] - df.index[0]).days / 365.25)) - 1
    sharpe = np.mean(df['Strategy Returns']) / np.std(df['Strategy Returns']) * np.sqrt(252)
    sortino = np.mean(df['Strategy Returns']) / np.std(df['Strategy Returns'][df['Strategy Returns'] < 0]) * np.sqrt(252)
    max_drawdown = ((df['Cumulative'].cummax() - df['Cumulative']).max() / df['Cumulative'].cummax().max())
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
    plt.plot(df.index, df['Cumulative'], label=strategy_name)
    plt.plot(df.index, (df['Adj Close'] / df['Adj Close'].iloc[0]), label='Buy and Hold')
    plt.title(f"Equity Curve for {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Returns")
    plt.legend()
    plt.show()

def todays_signal(ticker):
    """Download recent data and print current signal and position."""
    df = download_data(ticker, (datetime.now() - pd.Timedelta(days=RSI_PERIOD*2)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
    df = compute_signal(df)
    current_signal = df['Signal'].iloc[-1]
    position = "Long" if current_signal == 1 else "Short" if current_signal == -1 else "Neutral"
    print(f"Today's signal for {ticker}: {position}")

def main():
    parser = argparse.ArgumentParser(description='Fixed Fractional Position Sizing with Drawdown Cap Strategy')
    parser.add_argument('--ticker', type=str, default='SPY', help='Ticker symbol')
    parser.add_argument('--start', type=str, default='2010-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'), help='End date (YYYY-MM-DD)')
    parser.add_argument('--plot', action='store_true', help='Plot the equity curve')
    args = parser.parse_args()

    df = download_data(args.ticker, args.start, args.end)
    df = compute_signal(df)
    df = backtest(df)
    perf = performance(df)
    print_results(perf, args.ticker)

    if args.plot:
        plot_results(df, args.ticker, "Fixed Fractional Position Sizing with Drawdown Cap")

    todays_signal(args.ticker)

if __name__ == "__main__":
    main()
```