#!/usr/bin/env python
# coding: utf-8

import streamlit as st
# Set page configuration must be the first streamlit command
st.set_page_config(
    page_title="Technical Analysis Dashboard",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import time
import openai
import sys
from datetime import datetime
import os
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Union

# Load environment variables
load_dotenv()

sys.stdout.reconfigure(encoding='utf-8')

# Custom CSS for better appearance
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    /* Sidebar styling */
    .css-1d391kg {  /* Sidebar */
        background-color: #f8f9fa;
    }
    .css-1d391kg .block-container {
        padding-top: 2rem;
    }
    /* Ensure all text in sidebar is dark */
    .css-1d391kg, .css-1d391kg p, .css-1d391kg label, .css-1d391kg input {
        color: #212529 !important;
    }
    /* Style for text inputs in sidebar */
    .stTextInput input {
        color: #212529 !important;
        caret-color: #212529 !important;  /* Dark cursor color */
        background-color: white !important;  /* Ensure white background */
    }
    /* Make cursor visible on any background */
    input {
        caret-color: #212529 !important;  /* Dark cursor color */
    }
    /* Focus state for better visibility */
    .stTextInput input:focus {
        background-color: white !important;
        border-color: #80bdff !important;
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25) !important;
    }
    /* Chat message styling in sidebar */
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        background-color: #ffffff;
        color: #212529 !important;
    }
    .chat-message b {
        color: #212529 !important;
    }
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #0d6efd;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #f8f9fa;
    }
    /* Metric card styling */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1.2rem;
        margin: 0.5rem 0;
        text-align: center;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 0.8rem;
    }
    .metric-card h2 {
        font-size: 1.6rem;
        color: #212529;
        margin: 0;
        line-height: 1.2;
    }
    .metric-card p {
        margin: 0.8rem 0 0 0;
        font-weight: bold;
        font-size: 1.1rem;
    }
    /* Selectbox styling */
    .stSelectbox label {
        color: #212529 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ Function to format metrics
def format_metric_card(label, value, delta=None):
    """Create a formatted metric card with optional delta"""
    if delta:
        try:
            # Try to handle numerical delta (with percentage)
            delta_value = float(delta.replace('%', ''))
            color = "green" if delta_value > 0 else "red"
            delta_html = f'<div style="text-align: center; color: {color}; font-size: 0.9rem; margin-top: 0.5rem;">{delta}</div>'
        except ValueError:
            # Handle text-based delta (like "Neutral", "Overbought")
            color_map = {
                "Neutral": "gray",
                "Overbought": "red",
                "Oversold": "green"
            }
            color = color_map.get(delta, "gray")
            delta_html = f'<p style="color: {color}">({delta})</p>'
    else:
        delta_html = ''

    html = f"""
        <div class="metric-card">
            <h3>{label}</h3>
            <h2>{value}</h2>
            {delta_html}
        </div>
    """
    return html

# ✅ Cache chart generation
@st.cache_data(ttl=24*60*60)
def generate_charts(data, ticker, timeframe):
    """Generate all charts for a given stock and timeframe"""
    charts = {}
    
    # Price Chart
    fig_price, ax_price = plt.subplots(figsize=(10, 4))
    filtered_price = data['price'].tail(timeframe)
    ax_price.plot(filtered_price.index, filtered_price, label=f"{ticker} Price", color="blue")
    ax_price.set_xlabel("Date")
    ax_price.set_ylabel("Price ($)")
    ax_price.legend(loc='upper left', fontsize=10)
    ax_price.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    charts['price'] = fig_price
    
    # RSI Chart
    fig_rsi, ax_rsi = plt.subplots(figsize=(10, 4))
    filtered_rsi = data['rsi'].tail(timeframe)
    ax_rsi.plot(filtered_rsi.index, filtered_rsi, label="RSI", color="purple")
    ax_rsi.axhline(70, linestyle="dashed", color="red", label="Overbought (70)")
    ax_rsi.axhline(30, linestyle="dashed", color="green", label="Oversold (30)")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.set_xlabel("Date")
    ax_rsi.set_ylabel("RSI Value")
    ax_rsi.legend(loc='upper left', fontsize=10)
    ax_rsi.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    charts['rsi'] = fig_rsi
    
    # Moving Averages Chart
    fig_ma, ax_ma = plt.subplots(figsize=(10, 4))
    filtered_ma = data['ma'].tail(timeframe)
    ax_ma.plot(filtered_ma.index, filtered_ma[f'{ticker}_SMA20'], linestyle="dotted", color="orange", label="SMA 20")
    ax_ma.plot(filtered_ma.index, filtered_ma[f'{ticker}_SMA50'], linestyle="dashed", color="red", label="SMA 50")
    ax_ma.set_xlabel("Date")
    ax_ma.set_ylabel("Price ($)")
    ax_ma.legend(fontsize=10)
    ax_ma.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    charts['ma'] = fig_ma
    
    # MACD Chart
    fig_macd, ax_macd = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]}, figsize=(10, 6))
    filtered_macd = data['macd'].tail(timeframe)
    
    # MACD Line and Signal Line
    ax_macd[0].plot(filtered_macd.index, filtered_macd[f'{ticker}_MACD'], color='blue', linewidth=2, label='MACD')
    ax_macd[0].plot(filtered_macd.index, filtered_macd[f'{ticker}_Signal'], color='red', linestyle='--', linewidth=2, label='Signal')
    ax_macd[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax_macd[0].legend(loc='upper left', fontsize=10)
    ax_macd[0].grid(True, alpha=0.3)
    
    # MACD Histogram
    ax_macd[1].bar(filtered_macd.index, filtered_macd[f'{ticker}_Hist'],
                   color=np.where(filtered_macd[f'{ticker}_Hist'] > 0, 'green', 'red'))
    ax_macd[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax_macd[1].grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    charts['macd'] = fig_macd
    
    return charts

# Import functions from calc.py
from newcalc import (
    download_stock_data,
    calc_sharpe_ratio,
    calc_returns,
    calc_technical_indicators,
    ALL_TICKERS,
    TRADING_DAYS_YEAR
)

# Initialize data once at startup
@st.cache_data(ttl=24*60*60)
def initialize_data():
    """Initialize stock data with retries"""
    df = download_stock_data()
    if df.empty:
        st.error("Failed to download stock data. Using default data structure.")
        df = pd.DataFrame(index=pd.date_range(end=datetime.now(), periods=TRADING_DAYS_YEAR, freq='D'),
                         columns=['AAPL'] + ['AAPL_Daily_Return'])
    return df

# Initialize data once
df = initialize_data()

@st.cache_data(ttl=24*60*60)
def get_processed_stock_data(df: pd.DataFrame, ticker: str) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    """Process all stock data for a given ticker"""
    try:
        # Get technical indicators
        tech_data = calc_technical_indicators(df, ticker)
        
        # Get returns data
        returns = calc_returns(df)
        returns = returns[returns['Ticker'] == ticker]
        if returns.empty:
            returns = pd.DataFrame({'Ticker': [ticker], 'Return': [0.0]})
        
        # Get Sharpe ratio
        sharpe = calc_sharpe_ratio(df)
        sharpe = sharpe[sharpe['Ticker'] == ticker]
        if sharpe.empty:
            sharpe = pd.DataFrame({'Ticker': [ticker], 'Sharpe_Ratio': [0.0]})
        
        return {
            'price': tech_data['price'],
            'returns': returns,
            'sharpe': sharpe,
            'rsi': tech_data['rsi'],
            'ma': tech_data['ma'],
            'macd': tech_data['macd']
        }
    except Exception as e:
        st.error(f"Error processing data for {ticker}. Using default values.")
        empty_df = pd.DataFrame(index=pd.date_range(end=datetime.now(), periods=TRADING_DAYS_YEAR, freq='D'))
        return {
            'price': pd.Series(dtype='float64', index=empty_df.index),
            'returns': pd.DataFrame({'Ticker': [ticker], 'Return': [0.0]}),
            'sharpe': pd.DataFrame({'Ticker': [ticker], 'Sharpe_Ratio': [0.0]}),
            'rsi': pd.Series(50.0, index=empty_df.index),
            'ma': pd.DataFrame(columns=[f'{ticker}_SMA20', f'{ticker}_SMA50'], index=empty_df.index),
            'macd': pd.DataFrame(columns=[f'{ticker}_MACD', f'{ticker}_Signal', f'{ticker}_Hist'], index=empty_df.index)
        }

# Get available tickers
available_tickers = sorted(list(set([col for col in df.columns if not col.endswith('_Daily_Return')])))
Tickers = available_tickers if available_tickers else ['AAPL']  # Default to AAPL if no data

# ✅ Session State Management
if "selected_stock" not in st.session_state:
    st.session_state.selected_stock = "AAPL"
    st.session_state.previous_stock = None

def on_stock_change():
    """Callback function to handle stock selection changes"""
    if st.session_state.stock_selector != st.session_state.previous_stock:
        st.session_state.selected_stock = st.session_state.stock_selector
        st.session_state.previous_stock = st.session_state.stock_selector

# Configure OpenAI API key securely
try:
    # First try to get from environment variable
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
except Exception as e:
    st.error("OpenAI API key not found. Please set it in .env file or Streamlit secrets.")

# 📌 Enhanced Sidebar
with st.sidebar:
    st.warning(" :point_right: Not financial advice")
    st.image("graph.png", width=128)
    st.title("Dashboard Controls")
    
    st.subheader("📊 Select a Stock")
    selected_ticker = st.selectbox(
        "",
        Tickers,
        index=Tickers.index(st.session_state.selected_stock),
        key='stock_selector',
        on_change=on_stock_change
    )
    
    # Enhanced AI Q&A Section
    st.markdown("---")
    st.subheader("💡 Technical Analysis Guide")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Updated quick questions for technical indicators
    st.subheader(" :small_orange_diamond: Quick Questions")
    quick_questions = {
        "RSI Guide": "What is RSI (Relative Strength Index) and how is it used in technical analysis?",
        "MACD Guide": "What is MACD (Moving Average Convergence Divergence) and how should it be interpreted?",
        "Sharpe Ratio": "What is the Sharpe Ratio and why is it important for investment decisions?"
    }
    
    for label, question in quick_questions.items():
        if st.button(label):
            with st.spinner("Analyzing..."):
                # Modified AI response for general technical analysis questions
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a technical analysis expert. Provide clear, educational explanations about technical indicators and financial metrics. Include practical examples and interpretation guidelines."},
                        {"role": "user", "content": question}
                    ]
                )
                answer = response["choices"][0]["message"]["content"]
                st.session_state.chat_history.append((question, answer))
    
    # Custom question input for technical analysis
    st.markdown("---")
    st.subheader("🤖 AI Assistant")
    user_question = st.text_input(":memo: Ask anything about technical indicators:")
    if user_question and st.button("🤖 Ask AI"):
        with st.spinner("Analyzing..."):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a technical analysis expert. Provide clear, educational explanations about technical indicators and financial metrics. Include practical examples and interpretation guidelines."},
                    {"role": "user", "content": user_question}
                ]
            )
            answer = response["choices"][0]["message"]["content"]
            st.session_state.chat_history.append((user_question, answer))
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("Recent Analysis")
        for question, answer in reversed(st.session_state.chat_history[-3:]):  # Show last 3 conversations in reverse order
            st.markdown(f"""
                <div class="chat-message">
                    <b>Q: {question}</b><br>
                    A: {answer}
                </div>
            """, unsafe_allow_html=True)
        
        if st.button("Clear History"):
            st.session_state.chat_history = []

# Get processed data using the cached function
stock_data = get_processed_stock_data(df, selected_ticker)

# ✅ Enhanced Main Dashboard
st.title("📈 Technical Analysis Dashboard")
st.markdown(f"**Selected Stock: {selected_ticker}** | *Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

# Enhanced metrics display
metrics_container = st.container()
with metrics_container:
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate price difference percentage
    current_price = stock_data['price'].iloc[-1]
    yesterday_price = stock_data['price'].iloc[-2]
    price_change_pct = ((current_price - yesterday_price) / yesterday_price) * 100
    
    # Format metrics with better styling
    col1.markdown(format_metric_card(
        "Current Price",
        f"${current_price:,.2f}"
    ), unsafe_allow_html=True)
    col1.markdown(f'<div style="text-align: center; color: {"green" if price_change_pct > 0 else "red"}; font-size: 1.2rem; margin-top: -1rem;">{price_change_pct:+.2f}%</div>', unsafe_allow_html=True)
    
    yearly_return = stock_data['returns']['Return'].values[0] * 100
    col2.markdown(format_metric_card(
        "1-Year Return",
        f"{yearly_return:,.1f}%"
    ), unsafe_allow_html=True)
    
    col3.markdown(format_metric_card(
        "Sharpe Ratio",
        f"{stock_data['sharpe']['Sharpe_Ratio'].values[0]:,.2f}"
    ), unsafe_allow_html=True)
    
    rsi_value = stock_data['rsi'].iloc[-1]
    col4.markdown(format_metric_card(
        "RSI",
        f"{rsi_value:.1f}"
    ), unsafe_allow_html=True)

# Timeframe selection with better styling
st.markdown("---")
timeframe_col1, timeframe_col2 = st.columns([3, 1])
with timeframe_col2:
    timeframe_options = [30, 90, 180, 360]
    timeframe = st.selectbox(
        "📅 Timeframe",
        timeframe_options,
        key='timeframe',
        format_func=lambda x: f"{x} Days"
    )

# Generate and display cached charts
charts = generate_charts(stock_data, selected_ticker, timeframe)

# Display all charts with proper spacing
st.subheader(f"{selected_ticker} - Stock Price :chart_with_upwards_trend:")
st.pyplot(charts['price'])

st.subheader(f"{selected_ticker} - RSI :chart_with_upwards_trend:")
st.pyplot(charts['rsi'])

st.subheader(f"{selected_ticker} - Moving Averages :chart_with_upwards_trend:")
st.pyplot(charts['ma'])

st.subheader(f"{selected_ticker} - MACD Analysis :chart_with_upwards_trend:")
st.pyplot(charts['macd'])

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Data provided by Yahoo Finance</p>
        <p>Created by <a href="https://www.linkedin.com/in/ilias-roufogalis-320025347/" target="_blank" style="text-decoration: none; color: #0d6efd;">Ilias Roufogalis</a></p>
    </div>
""", unsafe_allow_html=True) 