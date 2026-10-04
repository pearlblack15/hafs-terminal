import pandas as pd
import numpy as np

def apply_simple_vol(df, cfg):
    vol_ma_len = int(cfg.get('vol_ma', 50))
    df['vol_ma'] = df['volume'].rolling(window=vol_ma_len, min_periods=1).mean()
    return df

def apply_vol_color_logic(df):
    # SPEED CHECK: If MTF Core already calculated the colors, skip the math instantly
    if 'vol_color' in df.columns:
        return df
        
    df['vol_sma_50'] = df['volume'].rolling(50, min_periods=1).mean()
    prev_close = df['close'].shift(1).fillna(df['open'])
    is_bull = df['close'] >= prev_close
    is_bear = df['close'] < prev_close
    
    # Calculate Max Down Volume for PPV (Pocket Pivot Volume) logic
    strict_dn = df['close'] < prev_close
    d_vol_col = df['volume'].where(strict_dn, 0.0)
    max_down_vol = d_vol_col.rolling(window=10, min_periods=1).max().shift(1).fillna(0)
    is_ppv = is_bull & (df['volume'] > max_down_vol)
    
    hvy = df['volume'].rolling(window=252, min_periods=1).max()
    hvq = df['volume'].rolling(window=63, min_periods=1).max()
    hve = df['volume'].cummax()
    
    vol_conditions = [
        (df['volume'] == hve),
        ((df['volume'] == hvy) & ~(df['volume'] == hve)),
        ((df['volume'] == hvq) & ~(df['volume'] == hve) & ~(df['volume'] == hvy)),
        (df['volume'] <= (df['vol_sma_50'] * 0.2)),
        is_ppv,
        is_bull & (df['volume'] > df['vol_sma_50']),
        is_bear & (df['volume'] > df['vol_sma_50'])
    ]
    vol_choices = ['#FF00FF', '#800080', '#800000', '#FFA500', '#0000FF', '#008000', '#FF0000']
    
    df['vol_color'] = np.select(vol_conditions, vol_choices, default='#3C3C3C')
    df['scaled_vol'] = np.where(df['vol_sma_50'] > 0, (df['volume'] / df['vol_sma_50']) * 15.0, 0.0)
    return df

def apply_custom_vol(df, cfg):
    return apply_vol_color_logic(df)

def apply_custom_vol_pane(df, cfg):
    return apply_vol_color_logic(df)

def apply_vol_rsi_pane(df, cfg):
    df = apply_vol_color_logic(df)
    
    # SPEED CHECK: Calculate RSI 2 ONLY if MTF Core is turned off
    if 'rsi_2' not in df.columns:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/2, min_periods=2).mean()
        avg_loss = loss.ewm(alpha=1/2, min_periods=2).mean()
        df['rsi_2'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        df['rsi_2'] = df['rsi_2'].fillna(50)
        
    return df