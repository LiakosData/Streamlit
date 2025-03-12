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
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

# Constants
RF_DAILY = 0.000177
RF_WEEKLY = 0.000841
RF_MONTHLY = 0.00368
TRADING_DAYS_YEAR = 252

# List of all tickers
ALL_TICKERS = [
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

@st.cache_data(ttl=24*60*60)
def get_session() -> requests.Session:
    """Create and configure a requests session with retries"""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[408, 429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=100)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache'
    }
    session.headers.update(headers)
    return session

@st.cache_data(ttl=24*60*60)
def download_chunk(tickers: List[str], session: requests.Session) -> Optional[pd.DataFrame]:
    """Download data for a chunk of tickers"""
    for attempt in range(3):
        try:
            df = yf.download(
                tickers,
                period='1y',
                progress=False,
                session=session,
                timeout=30
            )
            if not df.empty:
                return df['Close']
        except Exception:
            if attempt < 2:
                time.sleep(2)
    return None

@st.cache_data(ttl=24*60*60)
def download_stock_data() -> pd.DataFrame:
    """Download and process stock data with proper error handling"""
    session = get_session()
    chunk_size = 15
    df_list = []
    
    for i in range(0, len(ALL_TICKERS), chunk_size):
        chunk_tickers = ALL_TICKERS[i:i + chunk_size]
        chunk_df = download_chunk(chunk_tickers, session)
        if chunk_df is not None:
            df_list.append(chunk_df)
            time.sleep(1)
    
    if df_list:
        # Process data
        df = pd.concat(df_list, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Calculate daily returns
        returns_df = pd.DataFrame()
        for ticker in df.columns:
            returns_df[f'{ticker}_Daily_Return'] = df[ticker].pct_change().fillna(0)
        
        # Combine price and returns
        result = pd.concat([df, returns_df], axis=1)
        return result
    
    # Return empty DataFrame with correct structure
    return pd.DataFrame(index=pd.date_range(end=datetime.now(), periods=TRADING_DAYS_YEAR, freq='D'),
                       columns=ALL_TICKERS + [f'{t}_Daily_Return' for t in ALL_TICKERS])

@st.cache_data(ttl=24*60*60)
def get_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Get clean price data"""
    try:
        price_cols = [col for col in df.columns if not col.endswith('_Daily_Return')]
        return df[price_cols].fillna(method='ffill')
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=24*60*60)
def calc_sharpe_ratio(df: pd.DataFrame, rf: float = RF_DAILY) -> pd.DataFrame:
    """Calculate Sharpe Ratio with proper error handling"""
    if df.empty:
        return pd.DataFrame(columns=['Ticker', 'Sharpe_Ratio'])
    
    try:
        trading_days = max(len(df), 1)
        returns_data = df.filter(like='Daily_Return')
        
        if returns_data.empty:
            return pd.DataFrame(columns=['Ticker', 'Sharpe_Ratio'])
        
        results = []
        for col in returns_data.columns:
            ticker = col.replace('_Daily_Return', '')
            returns = returns_data[col]
            mean_return = returns.mean() or 0
            std_dev = returns.std() or 1
            
            sharpe = np.sqrt(trading_days) * ((mean_return - rf) / std_dev)
            results.append({'Ticker': ticker, 'Sharpe_Ratio': round(sharpe, 2)})
        
        return pd.DataFrame(results).sort_values('Sharpe_Ratio', ascending=False)
    except Exception:
        return pd.DataFrame(columns=['Ticker', 'Sharpe_Ratio'])

@st.cache_data(ttl=24*60*60)
def calc_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate yearly returns with proper error handling"""
    if df.empty:
        return pd.DataFrame(columns=['Ticker', 'Return'])
    
    try:
        returns_data = df.filter(like='Daily_Return')
        if returns_data.empty:
            return pd.DataFrame(columns=['Ticker', 'Return'])
        
        lookback_days = min(TRADING_DAYS_YEAR, len(returns_data))
        yearly_returns = (returns_data + 1).iloc[-lookback_days:].prod() - 1
        
        results = []
        for col in returns_data.columns:
            ticker = col.replace('_Daily_Return', '')
            ret = yearly_returns[col]
            results.append({'Ticker': ticker, 'Return': round(ret, 2)})
        
        return pd.DataFrame(results).sort_values('Return', ascending=False)
    except Exception:
        return pd.DataFrame(columns=['Ticker', 'Return'])

@st.cache_data(ttl=24*60*60)
def calc_technical_indicators(df: pd.DataFrame, ticker: str) -> Dict[str, pd.DataFrame]:
    """Calculate all technical indicators for a given ticker"""
    if df.empty or ticker not in df.columns:
        empty_df = pd.DataFrame(index=pd.date_range(end=datetime.now(), periods=TRADING_DAYS_YEAR, freq='D'))
        return {
            'price': pd.Series(dtype='float64', index=empty_df.index),
            'rsi': pd.Series(50.0, index=empty_df.index),
            'ma': pd.DataFrame(columns=[f'{ticker}_SMA20', f'{ticker}_SMA50'], index=empty_df.index),
            'macd': pd.DataFrame(columns=[f'{ticker}_MACD', f'{ticker}_Signal', f'{ticker}_Hist'], index=empty_df.index)
        }
    
    try:
        price_data = get_price_data(df)
        if price_data.empty:
            raise ValueError("No price data available")
        
        price = price_data[ticker]
        
        # Calculate RSI
        rsi = ta.momentum.RSIIndicator(price, window=14).rsi().fillna(50)
        
        # Calculate Moving Averages
        ma = pd.DataFrame(index=price.index)
        ma[f'{ticker}_SMA20'] = ta.trend.SMAIndicator(price, window=20).sma_indicator()
        ma[f'{ticker}_SMA50'] = ta.trend.SMAIndicator(price, window=50).sma_indicator()
        ma[f'{ticker}_EMA20'] = ta.trend.EMAIndicator(price, window=20).ema_indicator()
        ma[f'{ticker}_EMA50'] = ta.trend.EMAIndicator(price, window=50).ema_indicator()
        
        # Calculate MACD
        macd_ind = ta.trend.MACD(price)
        macd = pd.DataFrame(index=price.index)
        macd[f'{ticker}_MACD'] = macd_ind.macd()
        macd[f'{ticker}_Signal'] = macd_ind.macd_signal()
        macd[f'{ticker}_Hist'] = macd_ind.macd_diff()
        
        return {
            'price': price,
            'rsi': rsi,
            'ma': ma.fillna(method='ffill'),
            'macd': macd.fillna(0)
        }
    except Exception:
        empty_df = pd.DataFrame(index=pd.date_range(end=datetime.now(), periods=TRADING_DAYS_YEAR, freq='D'))
        return {
            'price': pd.Series(dtype='float64', index=empty_df.index),
            'rsi': pd.Series(50.0, index=empty_df.index),
            'ma': pd.DataFrame(columns=[f'{ticker}_SMA20', f'{ticker}_SMA50'], index=empty_df.index),
            'macd': pd.DataFrame(columns=[f'{ticker}_MACD', f'{ticker}_Signal', f'{ticker}_Hist'], index=empty_df.index)
        }

# Initialize data
df = download_stock_data()

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

def get_MA():
    if df.empty:
        return pd.DataFrame()
        
    try:
        df_MA = pd.DataFrame()
        price_data = get_price_data(df)
        
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
        price_data = get_price_data(df)
        
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