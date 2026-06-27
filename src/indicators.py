import pandas as pd
import numpy as np

def compute_rsi(series, period=14):
    """Calculates Relative Strength Index using standard rolling averages."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Calculate relative strength
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    """
    Engineers trend, momentum, and volatility indicators directly on the DataFrame.
    """
    df = df.copy().sort_values('Date').reset_index(drop=True)
    
    # 1. Trend Indicators (Moving Averages)
    df['SMA_Short'] = df['Close'].rolling(window=10).mean()
    df['SMA_Long'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # 2. Momentum Indicators (RSI & MACD)
    df['RSI_14'] = compute_rsi(df['Close'], period=14)
    df['MACD_Line'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
    
    # 3. Volatility Tracking (Bollinger Bands & ATR)
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Mid'] - (2 * df['BB_Std'])
    df['BB_pctB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift(1)).abs()
    low_close = (df['Low'] - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = tr.rolling(window=14).mean()
    
    return df

def generate_composite_signals(df):
    """
    Applies a weighted voting mechanic across indicators to generate consensus scores.
    Returns a directional layout matrix scaled between -1.0 and +1.0.
    """
    df = compute_indicators(df)
    
    # Standard internal voting weights assignment
    weights = {
        'ma_cross': 0.25,
        'rsi': 0.20,
        'macd': 0.20,
        'bollinger': 0.15,
        'trend_direction': 0.20
    }
    
    # Vectorized condition matrices
    vote_ma = np.where(df['SMA_Short'] > df['SMA_Long'], 1, -1)
    vote_rsi = np.where(df['RSI_14'] < 30, 1, np.where(df['RSI_14'] > 70, -1, 0))
    vote_macd = np.where(df['MACD_Hist'] > 0, 1, -1)
    vote_bb = np.where(df['Close'] < df['BB_Lower'], 1, np.where(df['Close'] > df['BB_Upper'], -1, 0))
    vote_trend = np.where(df['Close'] > df['BB_Mid'], 1, -1)
    
    # Map raw consensus sum array
    df['Composite_Score'] = (
        (vote_ma * weights['ma_cross']) +
        (vote_rsi * weights['rsi']) +
        (vote_macd * weights['macd']) +
        (vote_bb * weights['bollinger']) +
        (vote_trend * weights['trend_direction'])
    )
    
    # Convert numerical score range mapping into categorical action classes
    conditions = [
        (df['Composite_Score'] >= 0.4),
        (df['Composite_Score'] >= 0.15) & (df['Composite_Score'] < 0.4),
        (df['Composite_Score'] <= -0.4),
        (df['Composite_Score'] <= -0.15) & (df['Composite_Score'] > -0.4)
    ]
    choices = ['STRONG BUY', 'BUY', 'STRONG SELL', 'SELL']
    df['Signal'] = np.select(conditions, choices, default='HOLD')
    
    return df.dropna()

if __name__ == "__main__":
    # Internal module testing validation rule
    sample_df = pd.read_csv("data/raw/EUR_USD.csv")
    signals_df = generate_composite_signals(sample_df)
    print("=== Indicator Engine Verification Output ===")
    print(signals_df[['Date', 'Close', 'Composite_Score', 'Signal']].tail(10))