import yfinance as yf
import pandas 
from config import FETCH_DAYS

def fetch_historical_data(SYM = "BTC-USD"):
    print("Downloading historical data")
    data = yf.download(SYM, period=f"{FETCH_DAYS}d", interval="1h")
    print(data.head())   
    return data