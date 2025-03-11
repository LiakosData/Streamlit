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

def download_stock_data():
    """Download stock data using yfinance with proper headers and session configuration"""
    # Set up session with retries
    session = requests.Session()
    
    # More aggressive retry strategy
    retry_strategy = Retry(
        total=5,  # More retry attempts
        backoff_factor=1,  # Longer waits between retries
        status_forcelist=[408, 429, 500, 502, 503, 504],  # Retry on these status codes
        allowed_methods=["HEAD", "GET", "OPTIONS"]  # Allow retries on these methods
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=100)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    # Configure headers to mimic a browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Cache-Control': 'max-age=0'
    }
    session.headers.update(headers)

    # Configure yfinance session
    yf.shared._ERRORS = {}
    yf.shared._DFS = {}
    
    # Split tickers into smaller chunks to avoid rate limits
    all_tickers = [
        'NVDA', 'PLTR', 'TSLA', 'INTC', 'SMCI',
        'BAC',  'T',    'AAPL', 'AMZN', 'SPY',
        'HOOD', 'AMD',  'AVGO', 'MSFT', 'GOOG',
        'TSM',  'CSCO', 'WMT',  'BABA', 'WFC',
        'MU',   'MSTR', 'LRCX', 'MRVL', 'C',
        'DELL', 'XOM',  'META', 'V',    'ABBV',
        'VST',  'ORCL', 'JNJ',  'CRM',  'JPM',
        'PG',   'APP',  'COIN', 'TXN',  'IBM',
        'BRK-B','UNH',  'ACN',  'NFLX', 'SPOT',
        'LLY',  'ADBE', 'GS',   'COST', 'MA',
        'SOFI', 'PATH', 'CRSP', 'CELH', 'BBAI',
        'QCOM', 'SBUX', 'BP',   'KO',   'RGTI', 'SAP'
    ]
    
    chunk_size = 15  # Download in smaller chunks
    df_list = []
    
    try:
        for i in range(0, len(all_tickers), chunk_size):
            chunk_tickers = all_tickers[i:i + chunk_size]
            
            # Try multiple times for each chunk
            for attempt in range(3):  # 3 attempts per chunk
                try:
                    chunk_df = yf.download(
                        chunk_tickers,
                        period='1y',
                        progress=False,
                        session=session,
                        timeout=30  # Increased timeout
                    )
                    
                    if not chunk_df.empty:
                        df_list.append(chunk_df['Close'])
                        time.sleep(1)  # Rate limiting
                        break  # Success, move to next chunk
                except Exception as e:
                    if attempt == 2:  # Last attempt
                        print(f"Failed to download chunk {i//chunk_size + 1}: {str(e)}")
                    time.sleep(2)  # Wait before retry
                    continue
        
        if df_list:
            # Combine all chunks
            df = pd.concat(df_list, axis=1)
            df = df.loc[:, ~df.columns.duplicated()]  # Remove any duplicate columns
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate daily returns
            for ticker in df.columns:
                df[f'{ticker}_Daily_Return'] = df[ticker].pct_change().fillna(0)
            
            return df
            
    except Exception as e:
        print(f"Download failed: {str(e)}")
    
    # If everything fails, return empty DataFrame with correct structure
    empty_df = pd.DataFrame(columns=all_tickers)
    for ticker in all_tickers:
        empty_df[f'{ticker}_Daily_Return'] = pd.Series(dtype='float64')
    return empty_df

# Initialize data with retry
max_attempts = 3
for attempt in range(max_attempts):
    df = download_stock_data()
    if not df.empty and len(df) > 0:
        break
    time.sleep(5)  # Wait between attempts

# If still empty after all attempts, create empty DataFrame with correct structure
if df.empty:
    df = pd.DataFrame(columns=['AAPL'])  # Default to AAPL
    df[f'AAPL_Daily_Return'] = pd.Series(dtype='float64')

def calc_sharpe_ratio(df, rf_daily=0.0):
    """Calculate Sharpe Ratio for each stock in the DataFrame."""
    if df.empty:
        return pd.DataFrame()
        
    try:
        trading_days = max(len(df), 1)  # Avoid division by zero
        sharpe_ratios = pd.DataFrame(index=df.index)
        
        for ticker in df.columns:
            if not ticker.endswith('_Daily_Return'):
                daily_return_col = f'{ticker}_Daily_Return'
                if daily_return_col in df.columns:
                    mean_return = df[daily_return_col].mean() or 0
                    std_dev = df[daily_return_col].std() or 1  # Avoid division by zero
                    
                    sharpe_ratios[f'{ticker}_Sharpe_Ratio'] = np.sqrt(trading_days) * ((mean_return - rf_daily) / std_dev)
                    sharpe_ratios[f'{ticker}_Sharpe_Ratio'] = sharpe_ratios[f'{ticker}_Sharpe_Ratio'].round(2)
        
        return sharpe_ratios
    except:
        return pd.DataFrame()

def get_yearly_returns():
    if df.empty:
        return pd.DataFrame(columns=['Ticker', 'Return'])
        
    try:
        daily_returns = df.filter(like='Daily')
        if daily_returns.empty:
            return pd.DataFrame(columns=['Ticker', 'Return'])
            
        # Use last 252 days or all available days if less
        lookback_days = min(252, len(daily_returns))
        one_year_return = (daily_returns + 1).iloc[-lookback_days:].prod() - 1
        
        ret = pd.DataFrame(one_year_return)
        ret = ret.reset_index()
        ret.columns = ['Ticker', 'Return']
        ret['Ticker'] = ret['Ticker'].str.replace('_Daily_Return', '', regex=True)
        ret['Return'] = ret['Return'].round(2)
        return ret.sort_values(by='Return', ascending=False).reset_index(drop=True)
    except:
        return pd.DataFrame(columns=['Ticker', 'Return'])

def get_sharpe_ratio():
    if df.empty:
        return pd.DataFrame(columns=['Ticker', 'Sharpe_Ratio'])
        
    try:
        sharpe_df = calc_sharpe_ratio(df)
        if sharpe_df.empty:
            return pd.DataFrame(columns=['Ticker', 'Sharpe_Ratio'])
            
        sharp = sharpe_df.max()
        df_sharp = pd.DataFrame(sharp)
        df_sharp.index = df_sharp.index.str.replace('_Sharpe_Ratio', '', regex=True)
        df_sharp = df_sharp.rename(columns={0:'Sharpe_Ratio'})
        df_sharp = df_sharp.sort_values(by='Sharpe_Ratio', ascending=False)
        df_sharp = df_sharp.reset_index()
        df_sharp = df_sharp.rename(columns={'index':'Ticker'})
        return df_sharp
    except:
        return pd.DataFrame(columns=['Ticker', 'Sharpe_Ratio'])

def calc_rsi():
    if df.empty:
        return pd.DataFrame()
        
    try:
        df_rsi = pd.DataFrame()
        for ticker in df.columns:
            if not ticker.endswith('_Daily_Return'):
                rsi = ta.momentum.RSIIndicator(df[ticker], window=14).rsi()
                df_rsi[ticker] = rsi.fillna(50)  # Fill NaN with neutral RSI value
        return df_rsi
    except:
        return pd.DataFrame()

def get_price():
    if df.empty:
        return pd.DataFrame()
    try:
        return df[[col for col in df.columns if not col.endswith('_Daily_Return')]].fillna(method='ffill')
    except:
        return pd.DataFrame()

def get_MA():
    if df.empty:
        return pd.DataFrame()
        
    try:
        df_MA = pd.DataFrame()
        price_data = get_price()
        
        for ticker in price_data.columns:
            prices = price_data[ticker].fillna(method='ffill')
            df_MA[f'{ticker}_SMA20'] = ta.trend.SMAIndicator(prices, window=20).sma_indicator()
            df_MA[f'{ticker}_SMA50'] = ta.trend.SMAIndicator(prices, window=50).sma_indicator()
            df_MA[f'{ticker}_EMA20'] = ta.trend.EMAIndicator(prices, window=20).ema_indicator()
            df_MA[f'{ticker}_EMA50'] = ta.trend.EMAIndicator(prices, window=50).ema_indicator()
        
        df_MA.index = price_data.index
        return df_MA.fillna(method='ffill').round(2)
    except:
        return pd.DataFrame()

def get_MACD():
    if df.empty:
        return pd.DataFrame()
        
    try:
        df_MACD = pd.DataFrame()
        price_data = get_price()
        
        for ticker in price_data.columns:
            prices = price_data[ticker].fillna(method='ffill')
            macd = ta.trend.MACD(prices)
            df_MACD[f'{ticker}_MACD'] = macd.macd()
            df_MACD[f'{ticker}_Signal'] = macd.macd_signal()
            df_MACD[f'{ticker}_Hist'] = macd.macd_diff()
        
        df_MACD.index = price_data.index
        return df_MACD.fillna(0).round(2)
    except:
        return pd.DataFrame() 