import pandas as pd
import numpy as np

def ta_rma(series, length):
    return series.ewm(alpha=1/length, adjust=False).mean()

def ta_rsi(series, length):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    denom = ta_rma(loss, length).replace(0, np.nan)
    rs = ta_rma(gain, length) / denom
    return 100 - (100 / (1 + rs))

def ta_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def apply_standard_emas(df, cfg):
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    return df

def apply_ema_cross(df, cfg):
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['bull_cross'] = (df['ema_12'] > df['ema_26']) & (df['ema_12'].shift(1) <= df['ema_26'].shift(1))
    df['bear_cross'] = (df['ema_12'] < df['ema_26']) & (df['ema_12'].shift(1) >= df['ema_26'].shift(1))
    return df

def apply_macd(df, cfg):
    macd_f = int(cfg.get('macd_f', 12))
    macd_s = int(cfg.get('macd_s', 26))
    macd_sig = int(cfg.get('macd_sig', 9))
    
    df['macd'], df['macd_sig'], df['macd_hist'] = ta_macd(df['close'], fast=macd_f, slow=macd_s, signal=macd_sig)
    return df

def apply_rsi_14(df, cfg):
    rsi_len = int(cfg.get('rsi_len', 14))
    df['rsi_14'] = ta_rsi(df['close'], rsi_len)
    return df