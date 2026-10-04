import pandas as pd
import numpy as np
from . import loader, mtf_algo, standard_ta, volume_ta

# The Brains: Maps UI Checkboxes to the specific Python files
REGISTRY = {
    'mtf_core': mtf_algo.apply_mtf,
    'emas': standard_ta.apply_standard_emas,
    'ema_cross': standard_ta.apply_ema_cross,
    'macd': standard_ta.apply_macd,
    'rsi': standard_ta.apply_rsi_14,
    'vol_simple': volume_ta.apply_simple_vol,
    'vol_custom': volume_ta.apply_custom_vol,
    'vol_custom_pane': volume_ta.apply_custom_vol_pane,
    'vol_rsi': volume_ta.apply_vol_rsi_pane
}

def safe_float(val):
    if pd.isna(val) or np.isinf(val): return None
    return float(val)

def process_chart_data(filepath, full_name, tf='D', summary_only=False, active_inds=None, cfg=None):
    if active_inds is None: active_inds = ['mtf_core', 'vol_rsi']
    if cfg is None: cfg = {}

    # 1. Load and clean the CSV data
    df = loader.get_base_dataframe(filepath, tf)
    if df is None or df.empty: return []

    supply_zones = []

    # 2. Sequential Processing: Only runs the math for checked indicators
    for ind in active_inds:
        if ind in REGISTRY:
            if ind == 'mtf_core':
                df, supply_zones = REGISTRY[ind](df, cfg)
            else:
                df = REGISTRY[ind](df, cfg)

    # 3. Fast Exit for the Watchlist refresh
    if summary_only:
        if 'dash_signal' in df.columns and pd.notna(df['dash_signal'].iloc[-1]):
            return [{'dash_signal': str(df['dash_signal'].iloc[-1]), 'dash_sig_col': str(df['dash_sig_col'].iloc[-1])}]
        return [{'dash_signal': '-', 'dash_sig_col': '#d1d4dc'}]

    # 4. JSON Formatter
    records = df.to_dict('records')
    results = []

    for current in records:
        # RESTORED: Lightweight Charts strictly requires D/W/M data to be exactly 00:00 UTC. 
        # Using Asia/Kolkata shifts timestamps to 18:30, breaking chronological array syncs.
        try:
            ts = int(current['time'].tz_localize('UTC').timestamp() * 1000)
        except Exception:
            ts = int(current['time'].timestamp() * 1000)

        row_data = {
            'time': ts,
            'open': safe_float(current.get('open')),
            'high': safe_float(current.get('high')),
            'low': safe_float(current.get('low')),
            'close': safe_float(current.get('close')),
            'volume': safe_float(current.get('volume')),
            'full_name': full_name,
            
            # --- Dynamically inject standalone indicators (None if not calculated) ---
            'ema_9': safe_float(current.get('ema_9')), 
            'ema_21': safe_float(current.get('ema_21')),
            'ema_50': safe_float(current.get('ema_50')), 
            'ema_200': safe_float(current.get('ema_200')),
            'ema_12': safe_float(current.get('ema_12')),
            'ema_26': safe_float(current.get('ema_26')),
            
            # STRICT BOOLEAN SANITIZATION (Prevents NaN truthiness)
            'bull_cross': True if current.get('bull_cross') == True else False,
            'bear_cross': True if current.get('bear_cross') == True else False,
            
            'macd': safe_float(current.get('macd')),
            'macd_sig': safe_float(current.get('macd_sig')),
            'macd_hist': safe_float(current.get('macd_hist')),
            
            'rsi_14': safe_float(current.get('rsi_14')),
            'vol_ma': safe_float(current.get('vol_ma')),
            
            # --- MTF Core Outputs ---
            # STRICT STRING SANITIZATION (Prevents JSON breaking on Pandas NaN strings)
            'marker': current.get('marker') if pd.notna(current.get('marker')) else None, 
            'dash_signal': current.get('dash_signal') if pd.notna(current.get('dash_signal')) else '-', 
            'dash_sig_col': current.get('dash_sig_col') if pd.notna(current.get('dash_sig_col')) else '#d1d4dc',
            'vbcb_color': current.get('vbcb_color') if pd.notna(current.get('vbcb_color')) else None,
            'vol_color': current.get('vol_color') if pd.notna(current.get('vol_color')) else None,
            
            'scaled_vol': safe_float(current.get('scaled_vol')),
            'rsi_2': safe_float(current.get('rsi_2')),
            'bp_score': safe_float(current.get('bp_score')),
            'rvol': safe_float(current.get('rvol')),
            'ud_ratio': safe_float(current.get('ud_ratio')),
        }

        # Handle specific UI state variables for the MTF Engine
        if 'mtf_core' in active_inds:
            row_data['d_high'] = safe_float(current.get('d_high')) if current.get('d_active') == True else None
            row_data['w_high'] = safe_float(current.get('w_high')) if current.get('w_active') == True else None
            row_data['m_high'] = safe_float(current.get('m_high')) if current.get('m_active') == True else None
            row_data['trail_sl'] = safe_float(current.get('trail_sl')) if current.get('pos_state', 0) > 0 else None

        results.append(row_data)

    if len(results) > 0 and 'mtf_core' in active_inds:
        results[-1]['supply_zones'] = supply_zones
        
    return results