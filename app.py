import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import os
import numpy as np

from crypto_data import get_crypto_data
from technical_analysis import calculate_indicators
from news_sentiment import get_news, analyze_sentiment
from signal_generator import generate_signal
from utils import log_signal, get_logs
import database as db
import profit_analysis as pa

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Bot",
    page_icon="📈",
    layout="wide"
)

# Constants
SUPPORTED_CRYPTOS = ["BTC", "ETH", "BNB", "XRP", "ADA"]
TIMEFRAMES = ["1h", "4h", "1d"]

# App title and description
st.title("Cryptocurrency Trading Bot")
st.markdown("""
    This application combines technical analysis and news sentiment to generate trading signals for cryptocurrencies.
    *Note: This is a demo and does not execute real trades.*
""")

# Sidebar for settings
st.sidebar.header("Settings")

# Cryptocurrency selection
selected_crypto = st.sidebar.selectbox(
    "Select Cryptocurrency",
    SUPPORTED_CRYPTOS,
    index=0
)

# Timeframe selection
selected_timeframe = st.sidebar.selectbox(
    "Select Timeframe",
    TIMEFRAMES,
    index=2
)

# Period selection for historical data
days_lookback = st.sidebar.slider(
    "Days of Historical Data",
    min_value=7,
    max_value=90,
    value=30
)

# Trading settings section
st.sidebar.header("Trading Settings")

# Maximum trading percentage of available funds
max_trading_percentage = st.sidebar.slider(
    "Max Trading Percentage",
    min_value=5,
    max_value=100,
    value=50,
    help="Maximum percentage of available funds to use per trade"
)

# Stop loss and take profit settings
enable_auto_risk = st.sidebar.checkbox("Enable Auto Risk Management", value=True)
if not enable_auto_risk:
    stop_loss_pct = st.sidebar.slider(
        "Stop Loss (%)",
        min_value=1,
        max_value=15,
        value=5,
        help="Percentage below entry price to place stop loss"
    )
    take_profit_pct = st.sidebar.slider(
        "Take Profit (%)",
        min_value=1,
        max_value=30,
        value=10,
        help="Percentage above entry price to take profit"
    )

# Auto-refresh toggle and interval
auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=False)
refresh_interval = st.sidebar.number_input(
    "Refresh interval (minutes)",
    min_value=1,
    max_value=60,
    value=5,
    disabled=not auto_refresh
)

# Trigger for manual refresh
refresh_button = st.sidebar.button("Refresh Data")

# Display settings summary
st.sidebar.markdown("---")
st.sidebar.subheader("Current Settings")
st.sidebar.markdown(f"**Symbol:** {selected_crypto}/USDT")
st.sidebar.markdown(f"**Timeframe:** {selected_timeframe}")
st.sidebar.markdown(f"**Historical data:** {days_lookback} days")

# Placeholders for data
data_placeholder = st.empty()
tech_indicators_placeholder = st.empty()
news_sentiment_placeholder = st.empty()
signal_placeholder = st.empty()
history_placeholder = st.empty()

# Function to load and process data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data(symbol, timeframe, days):
    try:
        # Get cryptocurrency price data
        symbol_pair = f"{symbol}USDT"
        df = get_crypto_data(symbol_pair, timeframe, days)
        
        # Calculate technical indicators
        df = calculate_indicators(df)
        
        # Get news data and sentiment
        news_data = get_news(symbol)
        sentiment_score = analyze_sentiment(news_data)
        
        # Generate trading signal with risk management recommendations
        signal, confidence, explanation, stop_loss, take_profit = generate_signal(df, sentiment_score)
        
        # Log the signal to file (legacy)
        log_signal(symbol_pair, timeframe, signal, confidence, sentiment_score, explanation)
        
        # Store signal in database
        current_price = df['close'].iloc[-1] if not df.empty else 0
        db.store_trading_signal(
            symbol_pair, 
            timeframe, 
            signal, 
            confidence, 
            sentiment_score,
            stop_loss,
            take_profit,
            explanation,
            current_price
        )
        
        return df, news_data, sentiment_score, signal, confidence, explanation, stop_loss, take_profit
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, None, None, None, None, None

# Function to update the dashboard
def update_dashboard():
    with st.spinner("Loading data..."):
        df, news_data, sentiment_score, signal, confidence, explanation, stop_loss, take_profit = load_data(
            selected_crypto, selected_timeframe, days_lookback
        )
    
    if df is not None:
        # Display price chart
        with data_placeholder.container():
            st.subheader(f"{selected_crypto}/USDT Price Data")
            
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name="OHLC"
            )])
            
            # Add volume as bar chart
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['volume'],
                marker_color='rgba(128, 128, 128, 0.5)',
                name="Volume",
                yaxis="y2"
            ))
            
            # Add moving averages
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['ma20'],
                line=dict(color='blue', width=1),
                name="MA20"
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['ma50'],
                line=dict(color='orange', width=1),
                name="MA50"
            ))
            
            # Layout configuration
            fig.update_layout(
                title=f"{selected_crypto}/USDT - {selected_timeframe}",
                xaxis_title="Date",
                yaxis_title="Price (USDT)",
                xaxis_rangeslider_visible=False,
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right",
                    showgrid=False
                ),
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display technical indicators
        with tech_indicators_placeholder.container():
            st.subheader("Technical Indicators")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="RSI (14)",
                    value=f"{df['rsi'].iloc[-1]:.2f}",
                    delta=f"{df['rsi'].iloc[-1] - df['rsi'].iloc[-2]:.2f}"
                )
                
                # RSI interpretation
                rsi_value = df['rsi'].iloc[-1]
                if rsi_value < 30:
                    st.markdown("🟢 **Oversold** - Potential buy signal")
                elif rsi_value > 70:
                    st.markdown("🔴 **Overbought** - Potential sell signal")
                else:
                    st.markdown("⚪ **Neutral RSI**")
            
            with col2:
                macd = df['macd'].iloc[-1]
                macd_signal = df['macd_signal'].iloc[-1]
                macd_diff = macd - macd_signal
                
                st.metric(
                    label="MACD",
                    value=f"{macd:.4f}",
                    delta=f"{macd_diff:.4f}"
                )
                
                # MACD interpretation
                if macd > macd_signal:
                    st.markdown("🟢 **Bullish MACD** - MACD above signal line")
                else:
                    st.markdown("🔴 **Bearish MACD** - MACD below signal line")
            
            with col3:
                ma20 = df['ma20'].iloc[-1]
                ma50 = df['ma50'].iloc[-1]
                price = df['close'].iloc[-1]
                
                st.metric(
                    label="Price vs MA",
                    value=f"{price:.2f}",
                    delta=f"{(price / ma20 - 1) * 100:.2f}% from MA20"
                )
                
                # Price vs MAs interpretation
                if price > ma20 and price > ma50:
                    st.markdown("🟢 **Strong Uptrend** - Price above both MAs")
                elif price > ma20 and price < ma50:
                    st.markdown("🟡 **Weak Uptrend** - Price between MAs")
                else:
                    st.markdown("🔴 **Downtrend** - Price below both MAs")
        
        # Display news sentiment
        with news_sentiment_placeholder.container():
            st.subheader("News Sentiment Analysis")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                # Display sentiment score with a gauge
                if sentiment_score > 0.2:
                    sentiment_color = "green"
                    sentiment_label = "Positive"
                elif sentiment_score < -0.2:
                    sentiment_color = "red"
                    sentiment_label = "Negative"
                else:
                    sentiment_color = "gray"
                    sentiment_label = "Neutral"
                
                st.metric(
                    label="Sentiment Score",
                    value=f"{sentiment_score:.2f}",
                    delta=sentiment_label
                )
                
                # Create a simple gauge for sentiment
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sentiment_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [-1, 1]},
                        'bar': {'color': sentiment_color},
                        'steps': [
                            {'range': [-1, -0.2], 'color': "lightcoral"},
                            {'range': [-0.2, 0.2], 'color': "lightgray"},
                            {'range': [0.2, 1], 'color': "lightgreen"}
                        ]
                    },
                    title={'text': "Sentiment"}
                ))
                
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display recent news
                st.markdown("### Recent News")
                
                if news_data and len(news_data) > 0:
                    for i, news in enumerate(news_data[:3]):
                        st.markdown(f"**{news['title']}**")
                        st.markdown(f"*Source: {news['source']} - {news['published_at']}*")
                        st.markdown(f"{news['summary'][:200]}...")
                        if i < 2:
                            st.markdown("---")
                else:
                    st.info("No recent news available")
        
        # Display trading signal
        with signal_placeholder.container():
            st.subheader("Trading Signal")
            
            cols = st.columns([1, 2])
            
            with cols[0]:
                # Display the signal with appropriate color
                if signal == "BUY":
                    signal_color = "green"
                    signal_emoji = "🟢"
                elif signal == "SELL":
                    signal_color = "red"
                    signal_emoji = "🔴"
                else:  # HOLD
                    signal_color = "orange"
                    signal_emoji = "🟠"
                
                st.markdown(
                    f"<h2 style='color: {signal_color}; text-align: center;'>{signal_emoji} {signal}</h2>",
                    unsafe_allow_html=True
                )
                
                st.markdown(f"**Confidence:** {confidence:.2f}")
                st.markdown(f"**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            with cols[1]:
                # Display explanation for the signal
                st.markdown("### Signal Explanation")
                st.markdown(explanation)
                
                # Create a trading panel with max allocation, stop loss and take profit
                st.markdown("---")
                st.markdown("### Trade Execution Panel")
                
                # Get the latest price for calculations
                current_price = df['close'].iloc[-1]
                
                # Calculate max amount to trade based on user setting
                col1, col2 = st.columns(2)
                
                with col1:
                    # Simulated available balance
                    available_balance = 10000.00  # Example value in USDT
                    max_trade_amount = available_balance * (max_trading_percentage / 100)
                    
                    st.metric(
                        label="Available Balance (USDT)",
                        value=f"{available_balance:.2f}"
                    )
                    
                    st.metric(
                        label=f"Max Trade Amount ({max_trading_percentage}%)",
                        value=f"{max_trade_amount:.2f} USDT"
                    )
                    
                    # Calculate quantity based on current price
                    max_quantity = max_trade_amount / current_price
                    st.metric(
                        label=f"Max Quantity ({selected_crypto})",
                        value=f"{max_quantity:.5f}"
                    )
                
                with col2:
                    # Display AI-recommended risk management or user-selected values
                    if enable_auto_risk:
                        # Use AI-recommended values from signal generator
                        sl_value = stop_loss
                        tp_value = take_profit
                        st.success("Using AI-recommended risk management")
                    else:
                        # Use user-defined values from sidebar
                        sl_value = stop_loss_pct
                        tp_value = take_profit_pct
                        st.info("Using manual risk management settings")
                    
                    # Calculate stop loss and take profit prices
                    if signal == "BUY":
                        sl_price = current_price * (1 - sl_value/100)
                        tp_price = current_price * (1 + tp_value/100)
                    elif signal == "SELL":
                        sl_price = current_price * (1 + sl_value/100)
                        tp_price = current_price * (1 - tp_value/100)
                    else:
                        sl_price = current_price
                        tp_price = current_price
                    
                    # Display stop loss and take profit values
                    st.metric(
                        label="Stop Loss",
                        value=f"{sl_value:.1f}%",
                        delta=f"{sl_price:.2f} USDT"
                    )
                    
                    st.metric(
                        label="Take Profit",
                        value=f"{tp_value:.1f}%",
                        delta=f"{tp_price:.2f} USDT"
                    )
                
                # Execute trade button (for demonstration only)
                if signal == "BUY":
                    trade_button = st.button("Execute BUY Order", type="primary")
                    if trade_button:
                        st.success(f"Simulated BUY order placed for {max_quantity:.5f} {selected_crypto} at {current_price:.2f} USDT")
                elif signal == "SELL":
                    trade_button = st.button("Execute SELL Order", type="primary")
                    if trade_button:
                        st.success(f"Simulated SELL order placed for {max_quantity:.5f} {selected_crypto} at {current_price:.2f} USDT")
                else:
                    st.button("Execute Trade", disabled=True)
                    st.info("No trade signal currently active")
                
                # Add a disclaimer
                st.markdown("---")
                st.markdown("*Disclaimer: This is not financial advice. Always do your own research before trading.*")
        
        # Display trading history and database dashboard
        with history_placeholder.container():
            tab1, tab2, tab3, tab4 = st.tabs(["Trading Signals", "Executed Trades", "Account Performance", "Profit Analysis"])
            
            with tab1:
                st.subheader("Trading Signal History")
                
                # Get signals from database
                history_df = db.get_recent_signals(selected_crypto + "USDT")
                
                if not history_df.empty:
                    # Format the dataframe for display
                    display_df = history_df[['timestamp', 'signal', 'confidence', 'sentiment_score', 'stop_loss', 'take_profit']].copy()
                    display_df.columns = ['Timestamp', 'Signal', 'Confidence', 'Sentiment', 'Stop Loss %', 'Take Profit %']
                    display_df['Timestamp'] = pd.to_datetime(display_df['Timestamp'])
                    display_df = display_df.sort_values('Timestamp', ascending=False)
                    
                    # Add color coding for signals
                    def color_signal(val):
                        if val == 'BUY':
                            return 'background-color: #c6ecc6'
                        elif val == 'SELL':
                            return 'background-color: #ffcccb'
                        else:
                            return 'background-color: #ffe4b5'
                    
                    st.dataframe(
                        display_df.style.apply(lambda x: [color_signal(val) if i == 1 else '' for i, val in enumerate(x)], axis=1),
                        use_container_width=True
                    )
                else:
                    st.info("No trading signals available yet.")
            
            with tab2:
                st.subheader("Executed Trades")
                
                # Get executed trades
                trades_df = db.get_trade_history(selected_crypto + "USDT")
                
                if not trades_df.empty:
                    # Format the dataframe for display
                    display_df = trades_df[['timestamp', 'trade_type', 'quantity', 'price', 'total_value', 'status', 'profit_loss']].copy()
                    display_df.columns = ['Timestamp', 'Type', 'Quantity', 'Price', 'Total Value', 'Status', 'P/L']
                    display_df['Timestamp'] = pd.to_datetime(display_df['Timestamp'])
                    display_df = display_df.sort_values('Timestamp', ascending=False)
                    
                    # Add color coding for trade types and P/L
                    def color_trades(df):
                        styles = pd.DataFrame('', index=df.index, columns=df.columns)
                        
                        # Color trade type
                        styles.loc[df['Type'] == 'BUY', 'Type'] = 'background-color: #c6ecc6'
                        styles.loc[df['Type'] == 'SELL', 'Type'] = 'background-color: #ffcccb'
                        
                        # Color P/L
                        styles.loc[df['P/L'] > 0, 'P/L'] = 'background-color: #c6ecc6; color: darkgreen'
                        styles.loc[df['P/L'] < 0, 'P/L'] = 'background-color: #ffcccb; color: darkred'
                        
                        # Color status
                        styles.loc[df['Status'] == 'OPEN', 'Status'] = 'background-color: #f0f0f0; color: blue'
                        styles.loc[df['Status'] == 'CLOSED', 'Status'] = 'background-color: #e6e6e6'
                        
                        return styles
                    
                    st.dataframe(
                        display_df.style.apply(color_trades, axis=None),
                        use_container_width=True
                    )
                    
                    # Add demo trade buttons
                    st.markdown("#### Demo Trade Execution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Execute Demo BUY"):
                            # Get current price and calculate trade details
                            current_price = df['close'].iloc[-1] if not df.empty else 60000
                            quantity = 0.01  # Small fixed quantity for demo
                            total_value = current_price * quantity
                            
                            # Record the trade
                            trade_id = db.record_trade(
                                selected_crypto + "USDT",
                                "BUY",
                                quantity,
                                current_price,
                                total_value,
                                current_price * (1 - stop_loss/100),
                                current_price * (1 + take_profit/100)
                            )
                            
                            if trade_id:
                                st.success(f"Demo BUY executed! Bought {quantity} {selected_crypto} at {current_price} USDT")
                                st.rerun()  # Refresh to show new trade
                            else:
                                st.error("Failed to execute demo trade")
                    
                    with col2:
                        if st.button("Close Random Open Trade"):
                            # Get open trades
                            open_trades = db.get_open_trades()
                            
                            if not open_trades.empty:
                                # Get random open trade
                                trade = open_trades.iloc[0]
                                
                                # Simulate closing price with some profit/loss
                                import random
                                price_change = random.uniform(-0.05, 0.1)  # -5% to +10%
                                closing_price = trade['price'] * (1 + price_change)
                                
                                # Update trade status
                                if db.update_trade_status(trade['id'], 'CLOSED', closing_price):
                                    st.success(f"Closed trade #{trade['id']} with {'profit' if price_change > 0 else 'loss'}")
                                    st.rerun()
                                else:
                                    st.error("Failed to close trade")
                            else:
                                st.warning("No open trades to close")
                else:
                    st.info("No executed trades yet.")
                    
                    # Add demo trade button when no trades exist
                    if st.button("Create Demo Trade"):
                        # Get current price and calculate trade details
                        current_price = df['close'].iloc[-1] if not df.empty else 60000
                        quantity = 0.01  # Small fixed quantity for demo
                        total_value = current_price * quantity
                        
                        # Record the trade
                        trade_id = db.record_trade(
                            selected_crypto + "USDT",
                            "BUY",
                            quantity,
                            current_price,
                            total_value,
                            current_price * (1 - stop_loss/100),
                            current_price * (1 + take_profit/100)
                        )
                        
                        if trade_id:
                            st.success(f"Demo trade created! Bought {quantity} {selected_crypto} at {current_price} USDT")
                            st.rerun()  # Refresh to show new trade
                        else:
                            st.error("Failed to create demo trade")
            
            with tab3:
                st.subheader("Account Performance")
                
                # Get account data
                performance = db.get_account_performance()
                balance = db.get_current_balance()
                
                if performance and balance:
                    # Display account balance
                    cols = st.columns(3)
                    
                    cols[0].metric(
                        label="Account Balance",
                        value=f"{balance['balance']:.2f} USDT",
                        delta=f"{balance['balance'] - 10000:.2f}" if balance['balance'] != 10000 else None
                    )
                    
                    cols[1].metric(
                        label="Available",
                        value=f"{balance['available']:.2f} USDT"
                    )
                    
                    cols[2].metric(
                        label="Locked in Trades",
                        value=f"{balance['locked']:.2f} USDT"
                    )
                    
                    # Display performance metrics
                    st.markdown("#### Trading Performance")
                    
                    metric_cols = st.columns(4)
                    
                    metric_cols[0].metric(
                        label="Total Trades",
                        value=performance['total_trades']
                    )
                    
                    win_rate = performance['win_rate']
                    metric_cols[1].metric(
                        label="Win Rate",
                        value=f"{win_rate:.1f}%"
                    )
                    
                    metric_cols[2].metric(
                        label="Profit/Loss",
                        value=f"{performance['total_profit_loss']:.2f} USDT",
                        delta=f"{performance['total_profit_loss']:.2f}" if performance['total_profit_loss'] != 0 else None
                    )
                    
                    metric_cols[3].metric(
                        label="Avg. Trade P/L",
                        value=f"{performance['avg_profit_loss']:.2f} USDT"
                    )
                    
                    # Display win/loss stats
                    if performance['total_trades'] > 0:
                        st.markdown("#### Win/Loss Analysis")
                        
                        wl_cols = st.columns(2)
                        
                        wl_cols[0].metric(
                            label="Winning Trades",
                            value=performance['winning_trades'],
                            delta=f"{performance['winning_trades']/performance['total_trades']*100:.1f}%" if performance['total_trades'] > 0 else None
                        )
                        
                        wl_cols[1].metric(
                            label="Losing Trades",
                            value=performance['losing_trades'],
                            delta=f"{performance['losing_trades']/performance['total_trades']*100:.1f}%" if performance['total_trades'] > 0 else None,
                            delta_color="inverse"
                        )
                        
                        # Display average win and loss
                        avg_cols = st.columns(2)
                        
                        avg_cols[0].metric(
                            label="Avg. Win",
                            value=f"{performance['avg_winning_trade']:.2f} USDT" if performance['avg_winning_trade'] != 0 else "N/A"
                        )
                        
                        avg_cols[1].metric(
                            label="Avg. Loss",
                            value=f"{performance['avg_losing_trade']:.2f} USDT" if performance['avg_losing_trade'] != 0 else "N/A"
                        )
                else:
                    st.info("No trading performance data available yet.")
                    
                    # Option to initialize account
                    if st.button("Initialize Demo Account"):
                        if db.update_balance(10000.0, 10000.0, 0.0):
                            st.success("Demo account initialized with 10,000 USDT")
                            st.rerun()
                        else:
                            st.error("Failed to initialize account")
            
            with tab4:
                st.subheader("Potensi Profit dan Loss")
                
                # Calculate profit potential based on current signal
                if df is not None and not df.empty:
                    # Get profit potential metrics
                    profit_metrics = pa.calculate_profit_potential(
                        df, 
                        signal, 
                        confidence, 
                        stop_loss if 'stop_loss' in locals() else 5.0, 
                        take_profit if 'take_profit' in locals() else 10.0
                    )
                    
                    # Display key metrics
                    st.markdown("### Analisis Potensi Profit")
                    
                    # Top level metrics
                    metrics_cols = st.columns(3)
                    
                    with metrics_cols[0]:
                        st.metric(
                            label="Probabilitas Kemenangan",
                            value=f"{profit_metrics['win_probability']}%",
                            delta=f"{profit_metrics['win_probability'] - 50:.1f}%" if profit_metrics['win_probability'] != 50 else None
                        )
                    
                    with metrics_cols[1]:
                        st.metric(
                            label="Potensi Return",
                            value=f"{profit_metrics['expected_return']}%",
                            delta=f"{profit_metrics['expected_return']:.2f}%" if profit_metrics['expected_return'] != 0 else None
                        )
                    
                    with metrics_cols[2]:
                        st.metric(
                            label="Rasio Risk/Reward",
                            value=f"{profit_metrics['risk_reward_ratio']}",
                            delta="Baik" if profit_metrics['risk_reward_ratio'] >= 2 else "Perlu Perhatian"
                        )
                    
                    # Risk breakdown
                    st.markdown("### Potensi Untung dan Rugi")
                    risk_cols = st.columns(2)
                    
                    with risk_cols[0]:
                        st.metric(
                            label="Potensi Profit Maksimum",
                            value=f"+{profit_metrics['max_profit_potential']}%",
                            delta="Dari Take Profit"
                        )
                    
                    with risk_cols[1]:
                        st.metric(
                            label="Potensi Loss Maksimum",
                            value=f"-{profit_metrics['max_loss_potential']}%",
                            delta="Dari Stop Loss",
                            delta_color="inverse"
                        )
                    
                    # Run profit simulation
                    st.markdown("### Simulasi Monte Carlo")
                    
                    # Run simulation for current signal
                    sim_results = pa.run_profit_simulation(
                        df, 
                        signal, 
                        confidence, 
                        stop_loss if 'stop_loss' in locals() else 5.0, 
                        take_profit if 'take_profit' in locals() else 10.0,
                        10000.0,  # initial capital
                        0.1       # trading fee percentage
                    )
                    
                    # Display simulation results
                    sim_cols = st.columns(3)
                    
                    with sim_cols[0]:
                        st.metric(
                            label="Win Rate Simulasi",
                            value=f"{sim_results['win_rate']}%"
                        )
                    
                    with sim_cols[1]:
                        st.metric(
                            label="Rata-rata Profit",
                            value=f"{sim_results['average_profit']:.2f} USDT"
                        )
                    
                    with sim_cols[2]:
                        st.metric(
                            label="Rata-rata Loss",
                            value=f"{sim_results['average_loss']:.2f} USDT",
                            delta_color="inverse"
                        )
                    
                    # Display scenario analysis
                    scenario_cols = st.columns(3)
                    
                    with scenario_cols[0]:
                        st.metric(
                            label="Skenario Terburuk (5%)",
                            value=f"{sim_results['worst_case']:.2f} USDT",
                            delta=f"{sim_results['worst_case']/100:.2f}%" if sim_results['worst_case'] != 0 else None,
                        )
                    
                    with scenario_cols[1]:
                        st.metric(
                            label="Skenario Median",
                            value=f"{sim_results['median_case']:.2f} USDT",
                            delta=f"{sim_results['median_case']/100:.2f}%" if sim_results['median_case'] != 0 else None,
                        )
                    
                    with scenario_cols[2]:
                        st.metric(
                            label="Skenario Terbaik (95%)",
                            value=f"{sim_results['best_case']:.2f} USDT",
                            delta=f"{sim_results['best_case']/100:.2f}%" if sim_results['best_case'] != 0 else None,
                        )
                    
                    # Display probability distribution pie chart
                    st.markdown("### Distribusi Probabilitas Hasil")
                    
                    if len(sim_results['profit_scenarios']) > 0:
                        # Create pie chart of probability distribution
                        prob_dist = sim_results['probability_distribution']
                        labels = [
                            f"Loss Berat (>5%): {prob_dist['loss_severe']:.1f}%", 
                            f"Loss Ringan (<5%): {prob_dist['loss_moderate']:.1f}%", 
                            f"Profit Kecil (<3%): {prob_dist['profit_small']:.1f}%", 
                            f"Profit Baik (3-7%): {prob_dist['profit_good']:.1f}%", 
                            f"Profit Tinggi (>7%): {prob_dist['profit_excellent']:.1f}%"
                        ]
                        values = [
                            prob_dist['loss_severe'],
                            prob_dist['loss_moderate'],
                            prob_dist['profit_small'],
                            prob_dist['profit_good'],
                            prob_dist['profit_excellent']
                        ]
                        colors = ['#ff6666', '#ffb366', '#ffff99', '#b3ff66', '#66ff66']
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=labels,
                            values=values,
                            marker=dict(colors=colors),
                            textinfo='label+percent',
                            hole=0.4
                        )])
                        
                        fig.update_layout(
                            title_text="Distribusi Kemungkinan Hasil Trading",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Long-term projection
                    st.markdown("### Proyeksi Jangka Panjang (12 Bulan)")
                    
                    # Show projection inputs
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        initial_capital = st.number_input("Modal Awal (USDT)", min_value=1000.0, max_value=1000000.0, value=10000.0, step=1000.0)
                    with col2:
                        trades_per_month = st.number_input("Trades per Bulan", min_value=1, max_value=100, value=8, step=1)
                    with col3:
                        months = st.number_input("Jumlah Bulan", min_value=1, max_value=36, value=12, step=1)
                    
                    # Calculate projection
                    projection = pa.long_term_projection(
                        sim_results['win_rate'],
                        sim_results['average_profit'],
                        sim_results['average_loss'],
                        initial_capital,
                        trades_per_month,
                        months
                    )
                    
                    # Display projection summary
                    proj_cols = st.columns(3)
                    
                    with proj_cols[0]:
                        st.metric(
                            label="Modal Proyeksi Akhir",
                            value=f"{projection['projected_capital']:.2f} USDT",
                            delta=f"{projection['total_return_pct']:.2f}%" if projection['total_return_pct'] != 0 else None
                        )
                    
                    with proj_cols[1]:
                        st.metric(
                            label="Return Tahunan",
                            value=f"{projection['annualized_return_pct']:.2f}%"
                        )
                    
                    with proj_cols[2]:
                        st.metric(
                            label="Maximum Drawdown",
                            value=f"{projection['max_drawdown_pct']:.2f}%",
                            delta_color="inverse"
                        )
                    
                    # Display balance projection chart
                    if len(projection['monthly_balances']) > 1:
                        months_labels = [f"Bulan {i}" for i in range(months + 1)]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=months_labels,
                            y=projection['monthly_balances'],
                            mode='lines+markers',
                            name='Modal',
                            line=dict(color='royalblue', width=2)
                        ))
                        
                        fig.update_layout(
                            title='Proyeksi Modal 12 Bulan',
                            xaxis_title='Bulan',
                            yaxis_title='Modal (USDT)',
                            hovermode='x unified',
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display additional performance metrics
                    metrics_cols = st.columns(4)
                    
                    with metrics_cols[0]:
                        st.metric(
                            label="Total Trades",
                            value=f"{projection['expected_trades']}"
                        )
                    
                    with metrics_cols[1]:
                        st.metric(
                            label="Profit Factor",
                            value=f"{projection['profit_factor']}"
                        )
                    
                    with metrics_cols[2]:
                        st.metric(
                            label="Sharpe Ratio",
                            value=f"{projection['sharpe_ratio']}"
                        )
                    
                    with metrics_cols[3]:
                        st.metric(
                            label="% Trades Profit",
                            value=f"{sim_results['win_rate']}%"
                        )
                    
                    # Add disclaimer
                    st.info("**Disclaimer:** Proyeksi ini didasarkan pada simulasi statistik dan tidak menjamin hasil investasi di masa depan. Performa masa lalu tidak menunjukkan hasil di masa depan. Trading cryptocurrency memiliki risiko tinggi, termasuk kehilangan seluruh modal.")

# Initial load
update_dashboard()

# Setup auto-refresh if enabled
if auto_refresh:
    st.markdown("---")
    refresh_container = st.empty()
    
    with refresh_container:
        status = st.info(f"Auto-refreshing data every {refresh_interval} minutes. Next refresh in {refresh_interval} minutes.")
    
    while auto_refresh:
        time_left = refresh_interval * 60
        while time_left > 0:
            mins = time_left // 60
            secs = time_left % 60
            status.info(f"Auto-refreshing data every {refresh_interval} minutes. Next refresh in {mins}m {secs}s.")
            time.sleep(1)
            time_left -= 1
        
        status.info("Refreshing data...")
        update_dashboard()
        
# Handle manual refresh
if refresh_button:
    update_dashboard()
