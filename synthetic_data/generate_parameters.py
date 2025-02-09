import requests
from bs4 import BeautifulSoup

import yfinance as yf
import numpy as np
import scipy.stats
from random import randint
from arch import arch_model


def fetch_sp500_stocks():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    stocks = []
    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        symbol = cells[0].text.strip()
        name = cells[1].text.strip()
        stocks.append((symbol, name))
    return stocks

def fetch_historical_data(ticker, period='1y', interval='1d'):
    historical_data = yf.Ticker(ticker).history(period = period, interval = interval)
    
    return {
        "time": historical_data.index.strftime('%Y-%m-%d').tolist(),
        "price": historical_data['Close'].tolist()
    }

def fetch_strange_dates(data):
    if not data or "time" not in data or "price" not in data:
        raise ValueError("Data must contain 'time' and 'price' arrays")
    
    times = np.array([np.datetime64(t) for t in data["time"]])
    prices = np.array(data["price"], dtype=float)
    
    if len(prices) < 2:
        raise ValueError("Not enough price data to compute daily changes.")
    
    daily_changes = np.diff(prices)
    daily_dates = times[1:]
    
    # Filter out NaN or inf values
    valid_indices = np.isfinite(daily_changes)
    daily_changes = daily_changes[valid_indices]
    daily_dates = daily_dates[valid_indices]
    
    if len(daily_changes) == 0:
        raise ValueError("No valid daily changes after filtering NaN or inf values.")
    
    crit_value = scipy.stats.norm.ppf(1 - 0.05 / 2)
    
# Rescale daily_changes
    daily_changes = (daily_changes) * 10
    garch_fit = arch_model(daily_changes, vol='Garch', p=1, q=1).fit(disp='off')
    forecast = garch_fit.conditional_volatility
    
    mean = np.mean(daily_changes)
    stdev = np.std(daily_changes)
    
    mask = (np.abs(daily_changes) > (crit_value * forecast))
    mask = mask & ((np.abs(daily_changes - mean) / stdev) > crit_value)
    unusual_dates = daily_dates[mask]
    
    if unusual_dates.size == 0:
        raise Exception("No unusual dates found with combined tests")
    
    unusual_dates = np.sort(unusual_dates)
    gaps = np.diff(unusual_dates)
    gaps_in_days = gaps.astype('timedelta64[D]').astype(int)
    median_gap = np.median(gaps_in_days)
    gap_indices = np.where(gaps_in_days > median_gap)[0]
    
    if gap_indices.size == 0:
        ranges = [(unusual_dates[0], unusual_dates[-1])]
    else:
        start_indices = np.r_[0, gap_indices + 1]
        end_indices = np.r_[gap_indices, unusual_dates.size - 1]
        ranges = [(unusual_dates[s], unusual_dates[e]) for s, e in zip(start_indices, end_indices) if s != e]
    
    ranges.sort(key=lambda pair: (pair[1] - pair[0]).astype('timedelta64[D]').astype(int), reverse=True)
    
    formatted_ranges = [(str(start.astype('M8[D]')), str(end.astype('M8[D]'))) for start, end in ranges]
    
    max_date = times.max()
    adjusted_ranges = []
    for start_str, end_str in formatted_ranges:
        start_date = np.datetime64(start_str)
        end_date = np.datetime64(end_str)
        if start_date == end_date:
            if start_date < max_date:
                end_date = start_date + np.timedelta64(1, 'D')
            else:
                start_date = start_date - np.timedelta64(1, 'D')
        adjusted_ranges.append((str(start_date.astype('M8[D]')), str(end_date.astype('M8[D]'))))
    
    return adjusted_ranges


if __name__ == "__main__":
    sp500_stocks = fetch_sp500_stocks()
    for symbol, name in sp500_stocks:
        print(f"{symbol}: {name}")
    
    stock = sp500_stocks[randint(0, len(sp500_stocks) - 1)]
    ticker, name = stock[0], stock[1]
    strange_dates = fetch_strange_dates(fetch_historical_data(ticker))
    
    print(f"{ticker}: {name} has potential non-volatility swings during {strange_dates}")