import pandas as pd
import yfinance as yf
import numpy as np
import ta
import matplotlib.pyplot as plt
import streamlit as st
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

rf_daily = 0.000177
rf_weekly = 0.000841
rf_monthly = 0.00368

def prepare_data(df):
    """Prepare data for analysis"""
    # Create a copy and ensure datetime index
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    # Calculate daily returns for each ticker
    for ticker in df.columns:
        if f'{ticker}_Daily_Return' not in df.columns:
            df[f'{ticker}_Daily_Return'] = df[ticker].pct_change()
    
    # Drop NaN values and calculate Sharpe ratios
    df = df.dropna()
    df = calc_sharpe_ratio(df)
    return df

@st.cache_data(ttl=86400)
def download_stock_data():
    # Set up session with retries
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    # Configure headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    session.headers.update(headers)

    # Configure yfinance session
    yf.shared._ERRORS = {}
    yf.shared._DFS = {}

    Tickers = [
        'NVDA', 'PLTR', 'TSLA', 'INTC', 'SMCI',  # Line 1
        'BAC',  'T',    'AAPL', 'AMZN', 'SPY',   # Line 2
        'HOOD', 'AMD',  'AVGO', 'MSFT', 'GOOG',  # Line 3
        'TSM',  'CSCO', 'WMT',  'BABA', 'WFC',   # Line 4
        'MU',   'MSTR', 'LRCX', 'MRVL', 'C',     # Line 5
        'DELL', 'XOM',  'META', 'V',    'ABBV',  # Line 6
        'VST',  'ORCL', 'JNJ',  'CRM',  'JPM',   # Line 7
        'PG',   'APP',  'COIN', 'TXN',  'IBM',   # Line 8
        'BRK-B', 'UNH', 'ACN',  'NFLX', 'SPOT',  # Line 9
        'LLY',  'ADBE', 'GS',   'COST', 'MA' ,   # Line 10
        'SOFI', 'PATH', 'CRSP', 'CELH', 'BBAI',
        'JNJ',  'QCOM', 'SBUX',
        'SMCI', 'BP', 'ORCL', 'MSTR', 'KO',
        'CSCO', 'WMT', 'JNJ', 'MA', 'RGTI', 'SAP'   
    ]
   
    df = yf.download(Tickers, period='1y', progress=False, session=session)
    return df

df = download_stock_data()
time.sleep(2)
df.isnull().sum()
df = df['Close']

import warnings
warnings.filterwarnings('ignore')

Tickers = list(df.columns)

# Create a copy of the DataFrame to avoid modifying the original
df = df.copy()

# Ensure the index is datetime (do this once, outside the loop)
df.index = pd.to_datetime(df.index)

# Convert all ticker columns to numeric (do this once, outside the loop)
df[Tickers] = df[Tickers].apply(pd.to_numeric, errors='coerce')

# Calculate daily returns for all tickers at once
for ticker in Tickers:
    if f'{ticker}_Daily_Return' not in df.columns:
        # Add validation checks
        if ticker not in df.columns:
            st.error(f"Ticker {ticker} not found in the data")
            continue
        if df[ticker].empty:
            st.error(f"No data available for {ticker}")
            continue
        if df[ticker].isna().all():
            st.error(f"All values are NaN for {ticker}")
            continue
        
        try:
            df[f'{ticker}_Daily_Return'] = df[ticker].pct_change()
        except Exception as e:
            st.error(f"Error calculating daily returns for {ticker}: {str(e)}")
            continue

# Drop rows with NaN values (do this once, outside the loop)
df = df.dropna()

def calc_sharpe_ratio(df, rf_daily=0.0):
    """
    Calculate Sharpe Ratio for each stock in the DataFrame.
    """
    trading_days = len(df)
    sharpe_ratios = pd.DataFrame(index=df.index)
    
    for stock in df.columns:
        col_name = f"{stock}_Daily_Return"
        if col_name in df.columns:
            mean_return = df[col_name].mean()
            std_dev = df[col_name].std()
            
            if std_dev > 0:
                sharpe_ratios[f'{stock}_Sharpe_Ratio'] = np.sqrt(trading_days) * ((mean_return - rf_daily) / std_dev)
            else:
                sharpe_ratios[f'{stock}_Sharpe_Ratio'] = np.nan
            
            sharpe_ratios[f'{stock}_Sharpe_Ratio'] = sharpe_ratios[f'{stock}_Sharpe_Ratio'].round(2)
    
    df = pd.concat([df, sharpe_ratios], axis=1)
    return df

df = calc_sharpe_ratio(df)

def get_yearly_returns():
    for ticker in Tickers:
        one_year_return = (df.filter(like='Daily') + 1).iloc[-252:].prod() - 1
        ret = pd.DataFrame(one_year_return)
        ret = ret.reset_index().round(2)
        ret = ret.rename(columns={0:'Return'})
        ret = ret.rename(columns={'index': 'Ticker'})    
        ret['Ticker']=ret['Ticker'].str.replace('_Daily_Return', '', regex = True)
        df_returns = ret.sort_values(by='Return', ascending = False).reset_index(drop=True)
    return df_returns

def get_sharpe_ratio():
    sharp = df.filter(like='Shar').max()
    df_sharp = pd.DataFrame(sharp)
    df_sharp.index =df_sharp.index.str.replace('_Sharpe_Ratio', '', regex=True)
    df_sharp= df_sharp.rename(columns={0:'Sharpe_Ratio'})
    df_sharp = df_sharp.sort_values(by = 'Sharpe_Ratio', ascending = False)
    df_sharp = df_sharp.reset_index()
    df_sharp = df_sharp.rename(columns={'index':'Ticker'})
    return df_sharp

def calc_rsi():
    df_rsi = pd.DataFrame()
    for ticker in df.columns:
        if not ticker.endswith('_Daily_Return') and not ticker.endswith('_Sharpe_Ratio'):
            df_rsi[ticker] = ta.momentum.RSIIndicator(df[ticker], window=14).rsi()
    return df_rsi.dropna()

def get_price():
    df_price = df.iloc[:,:69]
    return df_price

def get_MA():
    df_MA = pd.DataFrame()
    for ticker in Tickers:
        df_MA[f'{ticker}_SMA20'] = ta.trend.SMAIndicator(df[ticker], window=20).sma_indicator()
        df_MA[f'{ticker}_SMA50'] = ta.trend.SMAIndicator(df[ticker], window=50).sma_indicator()
        df_MA[f'{ticker}_EMA20'] = ta.trend.EMAIndicator(df[ticker], window=20).ema_indicator()
        df_MA[f'{ticker}_EMA50'] = ta.trend.EMAIndicator(df[ticker], window=50).ema_indicator()
    df_MA.index = df.index.values
    df_MA = df_MA.dropna()
    return df_MA.round(2)

def get_MACD():
    df_MACD = pd.DataFrame()
    for ticker in Tickers:
        macd = ta.trend.MACD(df[ticker])
        df_MACD[f'{ticker}_MACD'] = macd.macd()
        df_MACD[f'{ticker}_Signal'] = macd.macd_signal()
        df_MACD[f'{ticker}_Hist'] = macd.macd_diff()
    df_MACD.index = df.index.values
    return df_MACD.round(2).dropna() 