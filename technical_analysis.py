import pandas as pd
import numpy as np

def calculate_indicators(df):
    """
    Calculate technical indicators for the given dataframe
    
    Parameters:
    df (pandas.DataFrame): DataFrame with OHLCV data
    
    Returns:
    pandas.DataFrame: Same DataFrame with additional columns for indicators
    """
    if df.empty:
        return df  # Return empty dataframe if input is empty
    
    # Make a copy to avoid modifying the original dataframe
    df_with_indicators = df.copy()
    
    # Simple Moving Averages
    df_with_indicators['ma20'] = df_with_indicators['close'].rolling(window=20).mean()
    df_with_indicators['ma50'] = df_with_indicators['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df_with_indicators['ema12'] = df_with_indicators['close'].ewm(span=12, adjust=False).mean()
    df_with_indicators['ema26'] = df_with_indicators['close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df_with_indicators['macd'] = df_with_indicators['ema12'] - df_with_indicators['ema26']
    df_with_indicators['macd_signal'] = df_with_indicators['macd'].ewm(span=9, adjust=False).mean()
    df_with_indicators['macd_hist'] = df_with_indicators['macd'] - df_with_indicators['macd_signal']
    
    # RSI (14-period)
    delta = df_with_indicators['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # Calculate RSI
    rs = avg_gain / avg_loss
    df_with_indicators['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20-period, 2 standard deviations)
    std_dev = df_with_indicators['close'].rolling(window=20).std()
    df_with_indicators['bb_middle'] = df_with_indicators['ma20']
    df_with_indicators['bb_upper'] = df_with_indicators['bb_middle'] + (std_dev * 2)
    df_with_indicators['bb_lower'] = df_with_indicators['bb_middle'] - (std_dev * 2)
    
    # ATR (Average True Range, 14-period)
    high_low = df_with_indicators['high'] - df_with_indicators['low']
    high_close = (df_with_indicators['high'] - df_with_indicators['close'].shift()).abs()
    low_close = (df_with_indicators['low'] - df_with_indicators['close'].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df_with_indicators['atr'] = tr.rolling(window=14).mean()
    
    # Stochastic Oscillator
    lowest_low = df_with_indicators['low'].rolling(window=14).min()
    highest_high = df_with_indicators['high'].rolling(window=14).max()
    
    k = 100 * ((df_with_indicators['close'] - lowest_low) / (highest_high - lowest_low))
    df_with_indicators['stoch_k'] = k
    df_with_indicators['stoch_d'] = k.rolling(window=3).mean()
    
    # Calculate support and resistance levels (using a simple method)
    # For a real system, more sophisticated methods would be better
    
    # Simple support: lowest low of last 10 periods
    df_with_indicators['support'] = df_with_indicators['low'].rolling(window=10).min()
    
    # Simple resistance: highest high of last 10 periods
    df_with_indicators['resistance'] = df_with_indicators['high'].rolling(window=10).max()
    
    # Fill NaN values in the first rows where calculations couldn't be made
    df_with_indicators.fillna(method='bfill', inplace=True)
    
    return df_with_indicators
