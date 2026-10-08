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

# ATR is bolted directly to the MTF engine so the trailing stop never breaks
def ta_atr(df, length):
    prev_close = df['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return ta_rma(tr, length)

def process_htf(df_tf):
    df_tf = df_tf.copy()
    if len(df_tf) == 0: return df_tf
    
    df_tf['rsi_2'] = ta_rsi(df_tf['close'], 2)
    dip, is_active, stored_high, sl_level = 0, False, np.nan, np.nan
    highs, sls, actives = [], [], []
    
    for i in range(len(df_tf)):
        current = df_tf.iloc[i]
        prev_rsi = df_tf['rsi_2'].iloc[i-1] if i > 0 else 50
        curr_rsi = current['rsi_2']
        
        if is_active and pd.notna(sl_level) and current['close'] < sl_level: is_active = False
        if curr_rsi < 30:
            dip += 1
            if is_active:
                if pd.isna(sl_level) or current['low'] > sl_level: sl_level = current['low']
                    
        rsi_cross_70 = (curr_rsi >= 70) and (prev_rsi < 70)
        trigger = rsi_cross_70 and (dip >= 2)
        if rsi_cross_70: dip = 0
            
        if trigger and not is_active:
            stored_high = current['high']
            sl_level = np.nan
            is_active = True
            
        highs.append(stored_high if is_active else np.nan)
        sls.append(sl_level if is_active else np.nan)
        actives.append(is_active)
        
    df_tf['anchor_high'] = highs
    df_tf['anchor_sl'] = sls
    df_tf['is_active'] = actives
    return df_tf

def apply_mtf(df, cfg):
    # 1. Base Logic (Self-contained calculation of dependencies)
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['vol_sma_50'] = df['volume'].rolling(50, min_periods=1).mean()
    df['atr_14'] = ta_atr(df, 14)
    df['struct_low'] = df['low'].rolling(5, min_periods=1).min()

    # 2. RSI 2 Engine
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/2, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, min_periods=2).mean()
    df['rsi_2'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    df['rsi_2'] = df['rsi_2'].fillna(50)

    # 3. VBCB Colors & Vol Footprint
    prev_close = df['close'].shift(1).fillna(df['open'])
    is_bull = df['close'] >= prev_close
    is_bear = df['close'] < prev_close

    vbcb_conditions = [
        is_bear & (df['volume'] > (df['vol_sma_50'] * 1.618)),
        is_bear & (df['volume'] >= (df['vol_sma_50'] * 0.618)),
        is_bear,
        is_bull & (df['volume'] > (df['vol_sma_50'] * 1.618)),
        is_bull & (df['volume'] >= (df['vol_sma_50'] * 0.618)),
        is_bull
    ]
    vbcb_choices = ['#8B0000', '#FF5252', '#FF9800', '#1B5E20', '#089981', '#00E676']
    df['vbcb_color'] = np.select(vbcb_conditions, vbcb_choices, default='#26a69a')

    df['vol_sma_safe'] = df['volume'].rolling(window=50, min_periods=1).mean()
    df['rvol'] = np.where(df['vol_sma_safe'] > 0, df['volume'] / df['vol_sma_safe'], 1.0)

    strict_up = df['close'] >= prev_close
    strict_dn = df['close'] < prev_close
    u_vol_col = df['volume'].where(strict_up, 0.0)
    d_vol_col = df['volume'].where(strict_dn, 0.0)
    sum_up_ud = u_vol_col.rolling(window=50, min_periods=1).sum()
    sum_dn_ud = d_vol_col.rolling(window=50, min_periods=1).sum()
    df['ud_ratio'] = np.where(sum_dn_ud == 0, 1.0, sum_up_ud / sum_dn_ud)
    max_down_vol = d_vol_col.rolling(window=10, min_periods=1).max().shift(1).fillna(0)

    # 4. Burst Power (BP) Engine
    candle_rng = df['high'] - df['low']
    cp = np.where(candle_rng != 0, (df['close'] - df['low']) / candle_rng, 0.0)
    pct_move = np.where(prev_close != 0, ((df['close'] - prev_close) / prev_close) * 100.0, 0.0)
    is_valid_day = (cp >= 0.50) & (candle_rng != 0) & (df['close'].shift(1).fillna(0) > 0)
    
    c5_cond = is_valid_day & (pct_move >= 4.98) & (pct_move < 9.99)
    c10_cond = is_valid_day & (pct_move >= 9.99) & (pct_move < 19.0)
    c19_cond = is_valid_day & (pct_move >= 19.0)
    
    cum_5 = c5_cond.astype(int).cumsum()
    cum_10 = c10_cond.astype(int).cumsum()
    cum_19 = c19_cond.astype(int).cumsum()
    
    lb_bars = 756
    c5_val = cum_5 - cum_5.shift(lb_bars).fillna(0)
    c10_val = cum_10 - cum_10.shift(lb_bars).fillna(0)
    c19_val = cum_19 - cum_19.shift(lb_bars).fillna(0)
    
    df['bp_score'] = np.round((c5_val / 5.0) + (c10_val / 2.0) + (c19_val / 0.5))

    # 5. Volume Anomaly & Trend Filters
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

    df['candle_range'] = df['high'] - df['low']
    df['close_percent'] = np.where(df['candle_range'] == 0, 0, (df['close'] - df['low']) / df['candle_range'])
    df['is_vol_anomaly'] = df['volume'] > (df['vol_sma_50'] * 1.5)
    df['is_strong_close'] = df['close_percent'] >= 0.50
    df['inst_backing'] = df['is_vol_anomaly'] & df['is_strong_close']
    df['recent_inst_backing'] = pd.Series(df['inst_backing']).rolling(5).sum().shift(1) > 0
    df['pass_entry_filter'] = df['inst_backing'] | df['recent_inst_backing']
    df['is_uptrend'] = (df['close'] > df['ema_21']) & (df['close'] > df['ema_50'])
    
    # Rubber Band & Exhaustion
    df['dist_to_21'] = np.where(df['ema_21'] > 0, ((df['close'] - df['ema_21']) / df['ema_21']) * 100, 0)
    df['close_10d_ago'] = df['close'].shift(10)
    df['runup_10d'] = np.where(df['close_10d_ago'] > 0, ((df['close'] - df['close_10d_ago']) / df['close_10d_ago']) * 100, 0)
    
    # Stockbee Coiled Spring
    df['hl_range'] = df['high'] - df['low']
    df['min_range_4'] = df['hl_range'].rolling(4, min_periods=1).min()
    df['is_nr4'] = (df['hl_range'] <= df['min_range_4']) & (df['hl_range'] > 0)
    df['is_inside_bar'] = (df['high'] < df['high'].shift(1)) & (df['low'] > df['low'].shift(1))
    df['is_vol_quiet'] = df['volume'] < (df['vol_sma_50'] * 0.85)
    df['is_coiled'] = (df['is_nr4'] | df['is_inside_bar']) & df['is_vol_quiet']
    
    df['is_rubber_band_snapped'] = df['dist_to_21'] > 10.0 
    df['is_exhausted'] = df['runup_10d'] > 15.0

    df['year_week'] = df['time'].dt.strftime('%G-%V')
    df['year_month'] = df['time'].dt.strftime('%Y-%m')
    
    w_raw = df.groupby('year_week').agg({'time':'first', 'open':'first', 'high':'max', 'low':'min', 'close':'last'})
    m_df = df.groupby('year_month').agg({'time':'first', 'open':'first', 'high':'max', 'low':'min', 'close':'last'})
    
    # 6. Weekly Supply Zones (Smart Base-Drop Detection)
    w_raw['body'] = (w_raw['close'] - w_raw['open']).abs()
    w_raw['range'] = w_raw['high'] - w_raw['low']
    
    # 1. Base Candle: Max 60% body (Filters out momentum spikes, keeps thick bases)
    w_raw['is_base'] = w_raw['body'] <= (w_raw['range'] * 0.60)
    
    # 2. The Drop: Next week must be a red candle closing below the base's low
    w_raw['next_close'] = w_raw['close'].shift(-1)
    w_raw['next_open'] = w_raw['open'].shift(-1)
    
    w_raw['is_drop'] = (w_raw['next_close'] < w_raw['next_open']) & \
                       (w_raw['next_close'] < w_raw['low'])
    
    # 3. Valid Supply Formation: A base immediately followed by a drop
    w_raw['is_supply_formation'] = w_raw['is_base'] & w_raw['is_drop']
    
    supply_zones = []
    current_close = df['close'].iloc[-1] if len(df) > 0 else 0
    formations = w_raw[w_raw['is_supply_formation']]
    
    for idx, row in formations.iloc[::-1].iterrows():
        loc_idx = w_raw.index.get_loc(idx)
        
        cluster_high = row['high']
        cluster_bot = min(row['open'], row['close'])
        cluster_time = row['time']  # <-- FIX: Track the actual start time!
        
        # Look backwards dynamically for consecutive base candles
        b_offset = 1
        while (loc_idx - b_offset) >= 0:
            prev_row = w_raw.iloc[loc_idx - b_offset]
            if prev_row['is_base']:
                cluster_high = max(cluster_high, prev_row['high'])
                cluster_bot = min(cluster_bot, prev_row['open'], prev_row['close'])
                cluster_time = prev_row['time']  # <-- FIX: Push drawing start point backwards
                b_offset += 1
            else:
                break  
        
        # Ensure the master cluster zone is overhead (active supply)
        if cluster_high > current_close:
            subsequent_closes = w_raw.loc[idx:]['close'].iloc[1:]
            # Ensure the zone hasn't been invalidated by a close above the highest wick
            if not (subsequent_closes > cluster_high).any():
                supply_zones.append({
                    'time': int(pd.to_datetime(cluster_time).tz_localize('UTC').timestamp() * 1000),
                    'top': float(cluster_high),
                    'bot': float(cluster_bot)
                })
        if len(supply_zones) >= 2:
            break
            
    w_df = process_htf(w_raw).shift(1)
    m_df = process_htf(m_df).shift(1)
    
    df['w_high'] = df['year_week'].map(w_df['anchor_high'])
    df['w_sl'] = df['year_week'].map(w_df['anchor_sl'])
    df['w_active'] = df['year_week'].map(w_df['is_active']).fillna(False)
    df['m_high'] = df['year_month'].map(m_df['anchor_high'])
    df['m_sl'] = df['year_month'].map(m_df['anchor_sl'])
    df['m_active'] = df['year_month'].map(m_df['is_active']).fillna(False)

    # 7. MTF RSI Vectorized Math
    w_diff = w_df['close'].diff()
    w_gain = w_diff.where(w_diff > 0, 0.0)
    w_loss = -w_diff.where(w_diff < 0, 0.0)
    w_rma_u = w_gain.ewm(alpha=1/2, min_periods=1, adjust=False).mean()
    w_rma_d = w_loss.ewm(alpha=1/2, min_periods=1, adjust=False).mean()
    
    df['w_rmau_prev'] = df['year_week'].map(w_rma_u).fillna(0)
    df['w_rmad_prev'] = df['year_week'].map(w_rma_d).fillna(0)
    df['w_close_prev'] = df['year_week'].map(w_df['close']).fillna(df['open'])
    
    w_change = df['close'] - df['w_close_prev']
    w_u = w_change.where(w_change > 0, 0.0)
    w_d = -w_change.where(w_change < 0, 0.0)
    
    w_rmau_forming = (df['w_rmau_prev'] * 1 + w_u) / 2.0
    w_rmad_forming = (df['w_rmad_prev'] * 1 + w_d) / 2.0
    df['w_rsi2'] = np.where(w_rmad_forming == 0, 100.0, 100.0 - (100.0 / (1.0 + (w_rmau_forming / w_rmad_forming))))
    
    m_diff = m_df['close'].diff()
    m_gain = m_diff.where(m_diff > 0, 0.0)
    m_loss = -m_diff.where(m_diff < 0, 0.0)
    m_rma_u = m_gain.ewm(alpha=1/2, min_periods=1, adjust=False).mean()
    m_rma_d = m_loss.ewm(alpha=1/2, min_periods=1, adjust=False).mean()
    
    df['m_rmau_prev'] = df['year_month'].map(m_rma_u).fillna(0)
    df['m_rmad_prev'] = df['year_month'].map(m_rma_d).fillna(0)
    df['m_close_prev'] = df['year_month'].map(m_df['close']).fillna(df['open'])
    
    m_change = df['close'] - df['m_close_prev']
    m_u = m_change.where(m_change > 0, 0.0)
    m_d = -m_change.where(m_change < 0, 0.0)
    
    m_rmau_forming = (df['m_rmau_prev'] * 1 + m_u) / 2.0
    m_rmad_forming = (df['m_rmad_prev'] * 1 + m_d) / 2.0
    df['m_rsi2'] = np.where(m_rmad_forming == 0, 100.0, 100.0 - (100.0 / (1.0 + (m_rmau_forming / m_rmad_forming))))

    df['mtf_rsi_valid'] = (df['w_rsi2'] >= 60.0) & (df['m_rsi2'] >= 60.0)

    # 8. STATE MACHINE LOOP SETUP
    records = df.to_dict('records')
    
    marker_col = [None] * len(records)
    dash_signal_col = ["-"] * len(records)
    dash_sig_col_col = ["#d1d4dc"] * len(records)
    trail_sl_col = [None] * len(records)
    pos_state_col = [0] * len(records)
    d_high_col = [None] * len(records)
    d_active_col = [False] * len(records)
    
    # State tracking variables
    d_dip, d_active, d_high, d_sl = 0, False, np.nan, np.nan
    pos_state, entry_price, trail_sl = 0, np.nan, np.nan
    breakout_high = np.nan
    
    last_traded_time, last_w_traded_time = None, None
    recoil_reset, is_anchor_dead = True, True
    sl_hit, atr_hit = False, False
    w_line_broken = False
    
    true_m_time, true_w_time, true_d_time = None, None, None
    prev_m_high, prev_w_high, prev_d_high = np.nan, np.nan, np.nan
    
    max_buy_ext, max_half_ext = 6.0, 10.0

    # 9. SIGNAL ENGINE (Iterates chronologically)
    for i in range(1, len(records)):
        current = records[i]
        prev = records[i-1]
        
        if d_active and pd.notna(d_sl) and current['close'] < d_sl: 
            d_active = False
        if current['rsi_2'] < 30:
            d_dip += 1
            if d_active:
                if pd.isna(d_sl) or current['low'] > d_sl: d_sl = current['low']
                
        rsi_cross_70 = (current['rsi_2'] >= 70) and (prev['rsi_2'] < 70)
        trigger = rsi_cross_70 and (d_dip >= 2)
        if rsi_cross_70: d_dip = 0
        
        new_d_line = False
        if trigger and not d_active: 
            d_high, d_sl, d_active = current['high'], np.nan, True
            new_d_line = True
            
        if current['m_active'] and (not prev['m_active'] or current['m_high'] != prev_m_high):
            true_m_time = current['time']
        if current['w_active'] and (not prev['w_active'] or current['w_high'] != prev_w_high):
            true_w_time = current['time']
        if new_d_line:
            true_d_time = current['time']
            
        prev_m_high, prev_w_high, prev_d_high = current['m_high'], current['w_high'], d_high
        
        is_time_seq = (true_m_time is not None and true_w_time is not None and true_d_time is not None) and (true_m_time <= true_w_time <= true_d_time)
        is_price_nest = (pd.notna(current['m_high']) and pd.notna(current['w_high']) and pd.notna(d_high)) and (d_high < current['w_high'] < current['m_high'])
        is_chronological_coil = d_active and current['w_active'] and current['m_active'] and is_time_seq and is_price_nest

        new_w_line = current['w_active'] and (not prev['w_active'] or current['w_high'] != prev_w_high)
        
        if new_w_line or new_d_line:
            w_line_broken = False
        elif current['w_active'] and current['high'] >= current['w_high']:
            w_line_broken = True

        active_target_line = np.nan
        anchor_lbl = "-"
        
        if d_active and pd.notna(d_high):
            active_target_line = d_high
            anchor_lbl = "D"
            
        if current['w_active'] and pd.notna(current['w_high']):
            if pd.isna(d_high):
                active_target_line = current['w_high']
                anchor_lbl = "W"
            elif current['w_high'] >= d_high:
                if current['close'] >= current['w_high']:
                    active_target_line = current['w_high']
                    anchor_lbl = "W"
                elif w_line_broken and current['close'] < current['w_high'] and current['close'] > d_high:
                    active_target_line = d_high
                    anchor_lbl = "D (W-Fail)"
                elif w_line_broken and current['close'] <= d_high:
                    active_target_line = d_high
                    anchor_lbl = "D"
        
        if pd.notna(active_target_line) and current['close'] < active_target_line:
            recoil_reset = True
            
        marker = None
        trigger_exit_1, trigger_exit_2 = False, False
        exit_reason = ""
        
        if pos_state > 0:
            new_trail = current['struct_low'] - (current['atr_14'] * 0.5)
            if pd.isna(trail_sl) or new_trail > trail_sl: trail_sl = new_trail
            
            cross_sl = pd.notna(d_sl) and current['close'] < d_sl
            cross_atr = current['close'] < trail_sl
            
            panic_exit = cross_sl and cross_atr
            if panic_exit:
                trigger_exit_2 = True
                pos_state = 0
                sl_hit, atr_hit = True, True
                trail_sl, entry_price = np.nan, np.nan
                last_traded_time = None
                marker = 'EXIT_2'
                exit_reason = "PANIC"
            else:
                if cross_sl and not sl_hit:
                    sl_hit = True
                    if pos_state == 2:
                        trigger_exit_1 = True
                        pos_state = 1
                        breakout_high = current['high']
                        marker = 'EXIT_1'
                        exit_reason = "DSL_HALF"
                    else:
                        trigger_exit_2 = True
                        pos_state = 0
                        trail_sl, entry_price = np.nan, np.nan
                        last_traded_time = None
                        marker = 'EXIT_2'
                        exit_reason = "DSL_FULL"
                        
                if cross_atr and not atr_hit and pos_state > 0:
                    atr_hit = True
                    if pos_state == 2:
                        trigger_exit_1 = True
                        pos_state = 1
                        breakout_high = current['high']
                        marker = 'EXIT_1'
                        exit_reason = "ATR_HALF"
                    else:
                        trigger_exit_2 = True
                        pos_state = 0
                        trail_sl, entry_price = np.nan, np.nan
                        last_traded_time = None
                        marker = 'EXIT_2'
                        exit_reason = "ATR_FULL"
        
        if trigger_exit_2:
            is_anchor_dead = True
            
        if new_w_line or new_d_line:
            is_anchor_dead = False
            
        is_above_d_line = d_active and pd.notna(d_high) and current['close'] > d_high
        is_above_w_line = current['w_active'] and pd.notna(current['w_high']) and current['close'] > current['w_high']
        
        is_fresh_d_line = (last_traded_time is None) or (true_d_time != last_traded_time)
        is_fresh_w_macro = (last_w_traded_time is None) or (true_w_time != last_w_traded_time)
        
        mtf_valid = current['mtf_rsi_valid']
        
        trigger_daily_entry = is_above_d_line and current['is_uptrend'] and mtf_valid and is_fresh_d_line and current['pass_entry_filter'] and pos_state == 0 and recoil_reset
        trigger_weekly_entry = is_above_w_line and current['is_uptrend'] and mtf_valid and is_fresh_w_macro and current['pass_entry_filter'] and pos_state == 0
        
        if trigger_weekly_entry:
            last_w_traded_time = true_w_time
            
        trigger_entry_1 = trigger_daily_entry or trigger_weekly_entry
        trigger_entry_2 = False
        
        if trigger_entry_1 and marker not in ['EXIT_1', 'EXIT_2']:
            pos_state = 1
            is_anchor_dead = False
            recoil_reset = False
            if trigger_daily_entry:
                last_traded_time = true_d_time
            entry_price = current['close']
            breakout_high = current['high']
            trail_sl = current['struct_low'] - (current['atr_14'] * 0.5)
            sl_hit, atr_hit = False, False
            marker = 'BUY_COIL' if is_chronological_coil else ('BUY_D' if trigger_daily_entry else 'BUY_W')
            
        is_standard_add = pd.notna(breakout_high) and current['close'] > breakout_high and current['close'] > active_target_line and current['is_uptrend'] and mtf_valid and current['pass_entry_filter']
        is_macro_add = trigger_weekly_entry and pos_state == 1 
        
        if pos_state == 1 and not trigger_entry_1 and (is_standard_add or is_macro_add):
            trigger_entry_2 = True
            pos_state = 2
            entry_price = current['close']
            breakout_high = np.nan
            atr_hit = False
            marker = 'ADD'
            if is_macro_add:
                last_w_traded_time = true_w_time
                
        if pos_state == 1:
            breakout_high = max(float(breakout_high) if pd.notna(breakout_high) else 0, current['high'])
            
        base_ext = ((current['close'] - active_target_line) / active_target_line * 100) if pd.notna(active_target_line) else np.nan
        
        near_9 = (current['low'] <= (current['ema_9'] * 1.025)) and (current['close'] >= (current['ema_9'] * 0.995))
        near_21 = (current['low'] <= (current['ema_21'] * 1.025)) and (current['close'] >= (current['ema_21'] * 0.995))
        zone_vol_ok = current['volume'] <= (current['vol_sma_50'] * 1.05)
        is_ema_zone_cond = (pos_state > 0) and pd.notna(base_ext) and (base_ext > max_buy_ext) and (near_9 or near_21) and zone_vol_ok

        sig_txt, sig_col = "-", "#d1d4dc"
        coil_suffix = " ⚡" if is_chronological_coil else ""
        any_trigger = trigger_daily_entry or trigger_weekly_entry
        
        if marker == 'EXIT_1':
            sig_txt, sig_col = "⚠️ HALF EXIT (DSL)" if "DSL" in exit_reason else "⚠️ HALF EXIT (ATR)", "#fb8c00"
        elif marker == 'EXIT_2':
            if "DSL" in exit_reason: sig_txt, sig_col = "💥 STOPPED (DSL)", "#f23645"
            elif "ATR" in exit_reason: sig_txt, sig_col = "💥 STOPPED (ATR)", "#f23645"
            else: sig_txt, sig_col = "💥 STOPPED (PANIC)", "#f23645"
        elif d_active or current['w_active']:
            if is_anchor_dead:
                pass
            elif any_trigger:
                prefix = "MACRO " if trigger_weekly_entry else ""
                if current.get('is_rubber_band_snapped', False):
                    sig_txt, sig_col = f"⚠️ {prefix}EXTENDED{coil_suffix}", "#ff9800" 
                elif current.get('is_exhausted', False):
                    sig_txt, sig_col = f"⚠️ {prefix}EXHAUSTED{coil_suffix}", "#ff9800"
                elif base_ext > max_half_ext: 
                    sig_txt, sig_col = f"🔴 {prefix}TOO HIGH{coil_suffix}", "#f23645"
                elif base_ext > max_buy_ext: 
                    sig_txt, sig_col = f"🟠 {prefix}BUY HALF{coil_suffix}", "#ff9800"
                else: 
                    sig_txt, sig_col = f"🟢 {prefix}BUY NOW{coil_suffix}", "#089981"
            elif trigger_entry_2:
                prefix = "MACRO " if is_macro_add else ""
                if current.get('is_rubber_band_snapped', False) or current.get('is_exhausted', False):
                    sig_txt, sig_col = f"⚠️ {prefix}ADD BLOCKED (EXT)", "#ff9800"
                else:
                    sig_txt, sig_col = f"🟢 {prefix}ADD POS", "#089981"
            else:
                if pos_state > 0:
                    if pd.notna(active_target_line) and current['close'] <= active_target_line and not (new_w_line or new_d_line):
                        wait_reentry = (pos_state == 1 and not is_fresh_d_line)
                        sig_txt, sig_col = ("⏳ WAIT RE-ENTRY", "#ff9800") if wait_reentry else ("⚠️ BASE BROKEN", "#ff9800")
                    else:
                        if base_ext > max_half_ext:
                            sig_txt, sig_col = ("🎯 AT EMA", "#2962ff") if is_ema_zone_cond else ("🔴 TOO HIGH", "#f23645")
                        elif base_ext <= max_buy_ext:
                            sig_txt, sig_col = "🟢 BUY PB", "#009688"
                        else:
                            sig_txt, sig_col = ("🎯 AT EMA", "#2962ff") if is_ema_zone_cond else ("🟠 BUY HALF", "#ff9800")
                else:
                    if pd.notna(active_target_line) and current['mtf_rsi_valid']:
                        if current['close'] > active_target_line:
                            sig_txt, sig_col = "🚫 FILTER FAIL", "#f23645"
                        else:
                            sig_txt, sig_col = ("⏳ PRIME COIL", "#FFB300") if current.get('is_coiled', False) else ("⏳ WAIT BO", "#ff9800")
        
        # Log the generated data back to our arrays
        marker_col[i] = marker
        dash_signal_col[i] = sig_txt
        dash_sig_col_col[i] = sig_col
        trail_sl_col[i] = trail_sl if pos_state > 0 else None
        pos_state_col[i] = pos_state
        d_high_col[i] = d_high
        d_active_col[i] = d_active
        
    # 10. Bind the arrays directly back into the Pandas DataFrame
    df['marker'] = marker_col
    df['dash_signal'] = dash_signal_col
    df['dash_sig_col'] = dash_sig_col_col
    df['trail_sl'] = trail_sl_col
    df['pos_state'] = pos_state_col
    df['d_high'] = d_high_col
    df['d_active'] = d_active_col

    return df, supply_zones
