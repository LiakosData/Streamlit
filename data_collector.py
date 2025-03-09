#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import yfinance as yf
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import sys
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# MySQL Configuration from environment variables
DB_CONFIG = {
    'host': os.getenv('MYSQL_HOST'),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': os.getenv('MYSQL_DATABASE')
}

# Validate database configuration
if not all(DB_CONFIG.values()):
    print("❌ Missing database configuration. Please check your .env file.")
    sys.exit(1)

def download_and_store_data():
    """Download stock data and store in MySQL database"""
    try:
        # Create SQLAlchemy engine
        engine = create_engine(
            f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        )

        # Define tickers
        tickers = [
            'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA',
            'SPY', 'QQQ', 'DIA',
            'TSLA', 'META', 'NFLX'
        ]
        
        print("Downloading stock data...")
        
        # Download all stocks at once
        df = yf.download(tickers, period='1y', progress=False)['Close']
        
        if not df.empty:
            # Prepare data for database
            df_melted = df.reset_index()
            df_melted = df_melted.melt(id_vars=['Date'], value_vars=tickers, var_name='ticker', value_name='price')
            df_melted['Date'] = pd.to_datetime(df_melted['Date']).dt.date
            
            # Store in database
            df_melted.to_sql('stock_prices', engine, if_exists='replace', index=False)
            print("✅ Data successfully stored in database!")
            
        else:
            print("❌ No data downloaded")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    download_and_store_data() 