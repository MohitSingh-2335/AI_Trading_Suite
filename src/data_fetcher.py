import yfinance as yf
import pandas as pd 
from config import ASSETS, FETCH_DAYS, LIVE_FETCH_LIMIT, LOOKBACK_WINDOW

def fetch_historical_data(BTC = "BTC-USD"):
    print("Downloading historical data")
    data = yf.download(BTC, period=f"{FETCH_DAYS}d", interval="1h")
    print(data.head())   
    return data