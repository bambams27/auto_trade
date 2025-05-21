import os
import pandas as pd
from datetime import datetime
import json
import csv

# Path for log storage
LOG_DIRECTORY = "logs"
SIGNALS_LOG = os.path.join(LOG_DIRECTORY, "trading_signals.csv")

def setup_log_directory():
    """Ensure the log directory exists"""
    if not os.path.exists(LOG_DIRECTORY):
        os.makedirs(LOG_DIRECTORY)

def log_signal(symbol, timeframe, signal, confidence, sentiment_score, explanation):
    """
    Log trading signal to CSV file
    
    Parameters:
    symbol (str): Trading pair symbol
    timeframe (str): Candlestick interval
    signal (str): Trading signal (BUY, SELL, HOLD)
    confidence (float): Signal confidence
    sentiment_score (float): News sentiment score
    explanation (str): Signal explanation
    """
    # Ensure log directory exists
    setup_log_directory()
    
    # Prepare log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "symbol": symbol,
        "timeframe": timeframe,
        "signal": signal,
        "confidence": confidence,
        "sentiment_score": sentiment_score,
        "explanation": explanation
    }
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(SIGNALS_LOG)
    
    # Write to CSV
    with open(SIGNALS_LOG, 'a', newline='') as f:
        fieldnames = log_entry.keys()
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(log_entry)

def get_logs(symbol=None, limit=20):
    """
    Retrieve trading signal logs
    
    Parameters:
    symbol (str, optional): Filter logs by symbol
    limit (int, optional): Maximum number of logs to return
    
    Returns:
    pandas.DataFrame: DataFrame with log entries
    """
    # Ensure log directory exists
    setup_log_directory()
    
    # Check if log file exists
    if not os.path.isfile(SIGNALS_LOG):
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=[
            "timestamp", "symbol", "timeframe", "signal", 
            "confidence", "sentiment_score", "explanation"
        ])
    
    # Read logs
    try:
        df = pd.read_csv(SIGNALS_LOG)
        
        # Filter by symbol if provided
        if symbol and not df.empty:
            df = df[df["symbol"] == symbol]
        
        # Sort by timestamp (latest first)
        if not df.empty:
            df = df.sort_values("timestamp", ascending=False)
        
        # Limit the number of entries
        if limit and not df.empty:
            df = df.head(limit)
        
        return df
    
    except Exception as e:
        print(f"Error reading logs: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error
