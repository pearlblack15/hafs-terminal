import pandas as pd
import numpy as np
#from flask import Flask, jsonify, request
from flask import Flask, jsonify, request, send_file
import traceback
import os
import yfinance as yf
from datetime import datetime, timedelta, date
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
import concurrent.futures
import json
import time

# Globals
ZERODHA_TOKENS = {}
TOKEN_FILE = "zerodha_tokens.json"
CACHE_EXPIRY_DAYS = 7  # Automatically update every 7 days

@app.route('/api/zerodha_tokens', methods=['GET'])
def get_zerodha_tokens():
    global ZERODHA_TOKENS
    
    # 1. RAM CACHE: Return instantly if already loaded
    if ZERODHA_TOKENS:
        return jsonify(ZERODHA_TOKENS)

    # 2. Check if we have a local file and how old it is
    file_exists = os.path.exists(TOKEN_FILE)
    file_is_old = False
    
    if file_exists:
        # Check file age in days
        file_age_seconds = time.time() - os.path.getmtime(TOKEN_FILE)
        if (file_age_seconds / 86400) > CACHE_EXPIRY_DAYS:
            file_is_old = True
            
    # 3. NETWORK FETCH: Triggered if file is missing OR older than 7 days
    if not file_exists or file_is_old:
        try:
            print("Checking for updated Zerodha tokens...")
            url = "https://api.kite.trade/instruments/NSE"
            df = pd.read_csv(url)
            fetched_tokens = pd.Series(df.instrument_token.values, index=df.tradingsymbol).to_dict()
            
            # Save the fresh copy (this automatically resets the file's age)
            with open(TOKEN_FILE, 'w') as f:
                json.dump(fetched_tokens, f)
                
            ZERODHA_TOKENS = fetched_tokens
            return jsonify(ZERODHA_TOKENS)
            
        except Exception as e:
            print(f"Could not fetch fresh tokens (Offline?). {e}")
            # If no internet, don't panic. Just let the code continue down to Step 4 
            # to use the old local file as a backup.

    # 4. DISK CACHE: Load local file if it's fresh (or if internet failed)
    if not ZERODHA_TOKENS and file_exists:
        try:
            with open(TOKEN_FILE, 'r') as f:
                ZERODHA_TOKENS = json.load(f)
        except Exception as e:
            print(f"Error reading local token file: {e}")

    return jsonify(ZERODHA_TOKENS)

def get_fast_cmp_and_chg(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [l for l in f.readlines() if l.strip()]
            if len(lines) < 3: return None, None
            
            header = lines[0].lower().split(',')
            close_idx = header.index('close') if 'close' in header else -1
            if close_idx == -1: return None, None
            
            prev_c = float(lines[-2].split(',')[close_idx])
            last_c = float(lines[-1].split(',')[close_idx])
            pct_chg = ((last_c - prev_c) / prev_c) * 100 if prev_c else 0
            
            return last_c, pct_chg
    except Exception:
        return None, None
# --- TARGETED RAM CACHE & COMPANY NAMES ---
RAM_CACHE = {}
COMPANY_NAMES = {}

try:
    # Build absolute path so Python always finds the file next to app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'watchlist_fullname.csv')
    
    # Reads the Chartink CSV once on boot and builds a lightning-fast dictionary
    df_wl = pd.read_csv(csv_path)
    for _, row in df_wl.iterrows():
        COMPANY_NAMES[str(row['Symbol']).strip().upper()] = str(row['Stock Name']).strip()
    print(f"✅ Loaded {len(COMPANY_NAMES)} company names into RAM.")
except Exception as e:
    print(f"⚠️ Notice: Could not load Chartink CSV - {e}")

from indicators.dispatcher import process_chart_data, safe_float

def update_symbol_csv(symbol):
    filepath = f'historical_data/{symbol}.csv'
    
    # --- ON-THE-FLY DOWNLOADER FOR NEW STOCKS ---
    if not os.path.exists(filepath):
        print(f"📥 Downloading full history for new stock: {symbol}...")
        # Download maximum available history since inception
        new_data = yf.download(f"{symbol}.NS", period="max", progress=False)
        if new_data.empty: 
            new_data = yf.download(f"{symbol}.BO", period="max", progress=False)
        if new_data.empty: 
            return {"status": "error", "message": "Not found on Yahoo"}
            
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = [str(c[0]).lower().strip() for c in new_data.columns]
        else:
            new_data.columns = [str(c).lower().strip() for c in new_data.columns]
            
        new_data.reset_index(inplace=True)
        for col in new_data.columns:
            if col in ['date', 'index', 'datetime']:
                new_data.rename(columns={col: 'Date'}, inplace=True)
                break
        new_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        if new_data['Date'].dt.tz is not None:
            new_data['Date'] = new_data['Date'].dt.tz_localize(None)
            
        cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        clean_df = new_data[[c for c in cols_to_keep if c in new_data.columns]].copy()
        clean_df.sort_values('Date', inplace=True)
        clean_df.to_csv(filepath, index=False)
        return {"status": "success"}

    # ... keep the rest of the existing function (df_existing = pd.read_csv...)
    df_existing = pd.read_csv(filepath)
    df_existing.columns = [str(c).lower().strip() for c in df_existing.columns]
    date_col = 'time' if 'time' in df_existing.columns else 'date'
    
    df_existing[date_col] = pd.to_datetime(df_existing[date_col])
    if df_existing[date_col].dt.tz is not None: 
        df_existing[date_col] = df_existing[date_col].dt.tz_convert(None)
        
    # Ensure it's treated as a date object
    last_recorded_date = pd.to_datetime(df_existing[date_col].max())

    # --- THE ULTIMATE INSTANT-SKIP LOGIC ---
    today = datetime.now()
    latest_market_day = pd.Timestamp.today()
    
    # 1. If it's a weekday but before 4:00 PM, today's EOD candle doesn't exist yet on Yahoo.
    # So the latest possible market data we can expect is from yesterday.
    if latest_market_day.weekday() < 5 and today.hour < 16:
        latest_market_day -= pd.Timedelta(days=1)
        
    # 2. Roll back over the weekend if necessary
    while latest_market_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        latest_market_day -= pd.Timedelta(days=1)

    # 3. If our CSV already has data up to the latest available market day, skip instantly!
    if last_recorded_date.date() >= latest_market_day.date():
        return {"status": "up_to_date"}

    # --- The Overlap Fetch (Only runs if we are actually missing days) ---
    safe_fetch_start = last_recorded_date - pd.Timedelta(days=4)
    start_fetch_str = safe_fetch_start.strftime('%Y-%m-%d')
    end_fetch_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Try NSE first, fallback to BSE for daily updates
    new_data = yf.download(f"{symbol}.NS", start=start_fetch_str, end=end_fetch_date, progress=False)
    if new_data.empty:
        new_data = yf.download(f"{symbol}.BO", start=start_fetch_str, end=end_fetch_date, progress=False)

    if new_data.empty: 
        return {"status": "up_to_date"}

    if isinstance(new_data.columns, pd.MultiIndex):
        new_data.columns = [str(c[0]).lower().strip() for c in new_data.columns]
    else:
        new_data.columns = [str(c).lower().strip() for c in new_data.columns]

    new_data.reset_index(inplace=True)
    for col in new_data.columns:
        if col in ['date', 'index', 'datetime']:
            new_data.rename(columns={col: 'Date'}, inplace=True)
            break
            
    new_data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    
    # Safely strip timezone without crashing if Yahoo returns naive data
    new_data['Date'] = pd.to_datetime(new_data['Date'])
    if new_data['Date'].dt.tz is not None:
        new_data['Date'] = new_data['Date'].dt.tz_localize(None)

    df_existing.rename(columns={date_col: 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    cols_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    merged_df = pd.concat([df_existing[cols_to_keep], new_data[cols_to_keep]], ignore_index=True)
    merged_df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
    merged_df.sort_values('Date', inplace=True)
    
    # --- THE FIX: Only save and report success if we actually added new rows ---
    if len(merged_df) <= len(df_existing):
        return {"status": "up_to_date"}
    
    merged_df.to_csv(filepath, index=False)
    
    # Invalidate Cache for this symbol
    keys_to_delete = [k for k in RAM_CACHE.keys() if k.startswith(symbol + "_")]
    for k in keys_to_delete: del RAM_CACHE[k]
        
    return {"status": "success"}


def get_lazy_intraday_data(symbol):
    """Phase 1: Lazy Fetcher - Self-healing and timezone aware."""
    os.makedirs('intraday_data', exist_ok=True)
    filepath = f'intraday_data/{symbol}_15m.csv'
    
    needs_fetch = True
    if os.path.exists(filepath):
        try:
            # Check if we already downloaded it today
            mtime = datetime.fromtimestamp(os.path.getmtime(filepath)).date()
            if mtime == date.today():
                needs_fetch = False
        except Exception:
            pass
            
    if needs_fetch:
        print(f"Fetching 60d 15m data for {symbol}...")
        
        # Try NSE first, fallback to BSE for intraday data
        new_data = yf.download(f"{symbol}.NS", interval='15m', period='60d', progress=False)
        if new_data.empty:
            new_data = yf.download(f"{symbol}.BO", interval='15m', period='60d', progress=False)
        
        if not new_data.empty:
            # Flatten MultiIndex if necessary
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = [str(c[0]).lower().strip() for c in new_data.columns]
            else:
                new_data.columns = [str(c).lower().strip() for c in new_data.columns]
            
            new_data.reset_index(inplace=True)
            
            # THE FIX: Force lowercase AGAIN to catch the 'Datetime' column created by reset_index
            new_data.columns = [str(c).lower().strip() for c in new_data.columns]
            
            for col in new_data.columns:
                if col in ['datetime', 'date', 'index']:
                    new_data.rename(columns={col: 'time'}, inplace=True)
                    break
                    
            # Normalize Timezone to pure IST before stripping
            if 'time' in new_data.columns and new_data['time'].dt.tz is not None:
                new_data['time'] = new_data['time'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
                
            new_data.to_csv(filepath, index=False)
        else:
            return pd.DataFrame()

    # Return cached intraday CSV (with Self-Healing for broken files)
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            df.columns = [str(c).lower().strip() for c in df.columns]
            
            if 'time' not in df.columns:
                raise ValueError("Corrupted CSV structure - Missing time column")
                
            df['time'] = pd.to_datetime(df['time'])
            return df
        except Exception as e:
            print(f"Corrupted cache detected for {symbol}. Deleting and refetching...")
            os.remove(filepath)
            return get_lazy_intraday_data(symbol) # Recursive self-heal
            
    return pd.DataFrame()

def crunch_intraday(symbol, tf='15m'):
    """Phase 2 & 3: NSE Resampling and Daily EMA Projection."""
    try:
        full_name = COMPANY_NAMES.get(symbol.upper(), symbol)
        df = get_lazy_intraday_data(symbol)
        if df.empty: return []

        # Phase 3: Exact NSE 75-minute grouping (Aligns bins starting precisely at 9:15 AM)
        if tf == '75m':
            # 555 minutes = 9:15 AM. Groups perfectly into 5 identical bins per session.
            df['group'] = df['time'].apply(lambda x: f"{x.date()} {(x.hour*60 + x.minute - 555) // 75}")
            agg_dict = {'time': 'first', 'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            df = df.groupby('group').agg(agg_dict).reset_index(drop=True)
            
        # Load Daily Data for EMA projection
        daily_filepath = f'historical_data/{symbol}.csv'
        if not os.path.exists(daily_filepath):
            return []
            
        daily_df = pd.read_csv(daily_filepath)
        daily_df.columns = [str(c).lower().strip() for c in daily_df.columns]
        date_col = 'time' if 'time' in daily_df.columns else 'date'
        daily_df['time'] = pd.to_datetime(daily_df[date_col])
        daily_df.sort_values('time', inplace=True)
        
        # Calculate True Daily EMAs
        daily_df['ema_9'] = daily_df['close'].ewm(span=9, adjust=False).mean()
        daily_df['ema_21'] = daily_df['close'].ewm(span=21, adjust=False).mean()
        daily_df['ema_50'] = daily_df['close'].ewm(span=50, adjust=False).mean()
        daily_df['ema_200'] = daily_df['close'].ewm(span=200, adjust=False).mean()
        
        daily_df['date'] = daily_df['time'].dt.normalize()
        daily_emas = daily_df[['date', 'ema_9', 'ema_21', 'ema_50', 'ema_200']].copy()
        
        # Phase 2: Merge the Daily EMAs back onto the Intraday candles based on matching calendar dates
        df['date'] = df['time'].dt.normalize()
        df = df.merge(daily_emas, on='date', how='left')
        
        # THE FIX: Fill missing EMA data forward, then backward to cover the edges
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        
        # Output format mimicking MTF structure so the frontend logic remains unified
        # --- Calculate RSI(2) and Scaled Volume for Intraday ---
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1/2, min_periods=2).mean()
        avg_loss = loss.ewm(alpha=1/2, min_periods=2).mean()
        df['rsi_2'] = 100 - (100 / (1 + (avg_gain / avg_loss)))
        df['rsi_2'] = df['rsi_2'].fillna(50)

        df['vol_sma_50'] = df['volume'].rolling(50, min_periods=1).mean()
        df['scaled_vol'] = np.where(df['vol_sma_50'] > 0, (df['volume'] / df['vol_sma_50']) * 30.0, 0.0)
        
        prev_close = df['close'].shift(1).fillna(df['open'])
        df['vol_color'] = np.where(df['close'] >= prev_close, '#089981', '#f23645')

        records = df.to_dict('records')
        results = []
        for current in records:
            results.append({
                'time': int(current['time'].tz_localize('Asia/Kolkata').timestamp() * 1000), 
                'open': safe_float(current.get('open')), 'high': safe_float(current.get('high')), 
                'low': safe_float(current.get('low')), 'close': safe_float(current.get('close')), 
                'volume': safe_float(current.get('volume')),
                'ema_9': safe_float(current.get('ema_9')), 'ema_21': safe_float(current.get('ema_21')),
                'ema_50': safe_float(current.get('ema_50')), 'ema_200': safe_float(current.get('ema_200')),
                'm_high': None, 'm_sl': None,'m_time': None, 'w_high': None, 'w_sl': None,'w_time': None,
                'd_high': None, 'd_sl': None, 'd_time': None, 'trail_sl': None,
                'marker': None, 'dash_signal': "-", 'dash_sig_col': "#d1d4dc",
                'dash_ext': "-", 'dash_pnl': "-", 'full_name': full_name, 'dash_target': "-",
                'vbcb_color': None, 'vol_color': current.get('vol_color'), 'scaled_vol': safe_float(current.get('scaled_vol')), 'rsi_2': safe_float(current.get('rsi_2'))
            })
        return results
    except Exception as e:
        print(f"Error Crunching Intraday {symbol}: {e}")
        traceback.print_exc()
        return []
        
@app.route('/api/stocks')
def get_stocks_list():
    if not os.path.exists('historical_data'): return jsonify([])
    files = sorted([f.replace('.csv', '') for f in os.listdir('historical_data') if f.endswith('.csv')])
    return jsonify(files)

@app.route('/api/watchlist', methods=['POST'])
def get_watchlist_summary():
    """Uses POST in chunks to safely grab CMP & Signals without browser URL freezing."""
    req_data = request.get_json()
    symbols = req_data.get('symbols', []) if req_data else []
    inds_list = req_data.get('inds', ['mtf_core']) if req_data else ['mtf_core']
    
    results = {}
    for symbol in symbols:
        filepath = f'historical_data/{symbol}.csv'
        if os.path.exists(filepath):
            try:
                inds_key = ",".join(sorted(inds_list))
                cache_key = f"{symbol}_D_{inds_key}"
                # Computes math only ONCE and holds the JSON string in Python RAM
                if cache_key not in RAM_CACHE:
                    full_name = COMPANY_NAMES.get(symbol.upper(), symbol)
                    calculated_data = process_chart_data(filepath, full_name, 'D', False, inds_list)
                    if len(calculated_data) > 0:
                        RAM_CACHE[cache_key] = calculated_data
                
                data = RAM_CACHE.get(cache_key, [])
                if len(data) >= 2:
                    last_c = float(data[-1]['close'])
                    prev_c = float(data[-2]['close'])
                    pct_chg = ((last_c - prev_c) / prev_c) * 100 if prev_c else 0
                    results[symbol] = {
                        'chg': pct_chg,
                        'cmp': last_c,
                        'signal': data[-1].get('dash_signal', '-'),
                        'bp': data[-1].get('bp_score', None),
                        'rvol': data[-1].get('rvol', None),
                        'ud': data[-1].get('ud_ratio', None)
                    }
            except Exception as e:
                pass
    return jsonify(results)

@app.route('/api/sync/<symbol>')
def sync_single_symbol(symbol):
    try:
        res = update_symbol_csv(symbol)
        return jsonify(res)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/data/<symbol>')
def get_chart_data(symbol):
    try:
        tf = request.args.get('tf', 'D')
        inds_str = request.args.get('inds', 'mtf_core,vol_rsi')
        cfg_str = request.args.get('cfg', '{}')
        
        inds_list = sorted([i.strip() for i in inds_str.split(',') if i.strip()])
        inds_key = ",".join(inds_list)
        
        cache_key = f"{symbol}_{tf}_{inds_key}_{cfg_str}"
        
        # Load instantly from RAM Cache if available (0ms delay)
        if cache_key in RAM_CACHE:
            return jsonify(RAM_CACHE[cache_key])
            
        # Route to specific cruncher based on requested timeframe
        if tf in ['15m', '75m']:
            data = crunch_intraday(symbol, tf)
        else:
            filepath = f'historical_data/{symbol}.csv'
            if not os.path.exists(filepath): 
                # Trigger the auto-downloader if the user clicked a brand new stock
                res = update_symbol_csv(symbol)
                if res.get("status") == "error":
                    return jsonify({"error": "Stock not found"}), 404
            full_name = COMPANY_NAMES.get(symbol.upper(), symbol)
            try:
                cfg_dict = json.loads(cfg_str)
            except:
                cfg_dict = {}
                
            data = process_chart_data(filepath, full_name, tf, False, inds_list, cfg_dict)
            
        if not data:
            return jsonify({"error": "Empty or corrupted dataset"}), 404
            
        RAM_CACHE[cache_key] = data
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/')
def serve_ui():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, 'index.html')
    return send_file(html_path)

if __name__ == '__main__':
    import webview
    import threading
    import os
    import sys
    import webbrowser
    import time

    if getattr(sys, 'frozen', False):
        # --- 1. PRODUCTION MODE (The compiled .exe for others) ---
        server_thread = threading.Thread(target=app.run, kwargs={'port': 5000, 'use_reloader': False})
        server_thread.daemon = True
        server_thread.start()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(base_dir, 'icon.ico')

        webview.create_window('Hafs Terminal', 'http://127.0.0.1:5000', width=1320, height=700, x=20, y=20, icon=icon_path)
        webview.start(private_mode=False)
        
    else:
        # --- 2. DEVELOPMENT MODE (For you running in the browser) ---
        def open_browser():
            time.sleep(1.5)  # Give Flask a second to boot up
            webbrowser.open('http://127.0.0.1:5000')
            
        # Opens your default browser to a single tab
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Disabled the reloader to prevent the double-tab glitch
        # Removed host='0.0.0.0' since mobile viewing is no longer needed
        # app.run(port=5000, use_reloader=False)
        app.run(host='0.0.0.0', port=5000, use_reloader=False)