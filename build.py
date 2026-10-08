import os
import json
import time
import pandas as pd
import yfinance as yf
from indicators.dispatcher import process_chart_data

def build_static_api():
    print("🚀 Starting Daily Static Build...")
    os.makedirs('public/api/data', exist_ok=True)
    os.makedirs('temp_historical_data', exist_ok=True)

    # 1. Fetch Public Zerodha Tokens
    print("Fetching Zerodha Tokens...")
    try:
        url = "https://api.kite.trade/instruments/NSE"
        df_tokens = pd.read_csv(url)
        fetched_tokens = pd.Series(df_tokens.instrument_token.values, index=df_tokens.tradingsymbol).to_dict()
        with open('public/api/zerodha_tokens.json', 'w') as f:
            json.dump(fetched_tokens, f)
    except Exception as e:
        print(f"⚠️ Failed to fetch tokens: {e}")

    # 2. Load Watchlist
    COMPANY_NAMES = {}
    try:
        df_wl = pd.read_csv('watchlist_fullname.csv')
        for _, row in df_wl.iterrows():
            COMPANY_NAMES[str(row['Symbol']).strip().upper()] = str(row['Stock Name']).strip()
    except Exception as e:
        print("❌ No watchlist_fullname.csv found!")
        return

    summary_data = {}
    # We bake ALL indicators into the JSON so you can toggle them freely on mobile
    all_inds = ['mtf_core', 'emas', 'ema_cross', 'macd', 'rsi', 'vol_simple', 'vol_custom', 'vol_custom_pane', 'vol_rsi']
    default_cfg = {'rsi_len': 14, 'macd_f': 10, 'macd_s': 26, 'macd_sig': 9, 'vol_ma': 50}

    # 3. Process each stock
    for symbol in COMPANY_NAMES.keys():
        print(f"Crunching {symbol}...")
        try:
            # Download exactly 5 years of daily data
            new_data = yf.download(f"{symbol}.NS", period="5y", progress=False)
            if new_data.empty: 
                new_data = yf.download(f"{symbol}.BO", period="5y", progress=False)
            if new_data.empty:
                continue

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

            filepath = f'temp_historical_data/{symbol}.csv'
            new_data.to_csv(filepath, index=False)

            # Run the MTF Core Math for Daily, Weekly, and Monthly
            full_name = COMPANY_NAMES.get(symbol, symbol)
            
            for tf in ['D', 'W', 'M']:
                chart_data = process_chart_data(filepath, full_name, tf, False, all_inds, default_cfg)
                
                # Save a separate JSON for each timeframe (e.g., RELIANCE_D.json, RELIANCE_W.json)
                if chart_data:
                    with open(f'public/api/data/{symbol}_{tf}.json', 'w') as f:
                        json.dump(chart_data, f)
                        
                    # Only extract the Watchlist Summary data from the Daily ('D') timeframe
                    if tf == 'D' and len(chart_data) >= 2:
                        last_c = chart_data[-1].get('close', 0)
                        prev_c = chart_data[-2].get('close', 0)
                        pct_chg = ((last_c - prev_c) / prev_c) * 100 if prev_c else 0
                        
                        summary_data[symbol] = {
                            'chg': pct_chg,
                            'cmp': last_c,
                            'signal': chart_data[-1].get('dash_signal', '-'),
                            'bp': chart_data[-1].get('bp_score', None),
                            'rvol': chart_data[-1].get('rvol', None),
                            'ud': chart_data[-1].get('ud_ratio', None)
                        }
        except Exception as e:
            print(f"❌ Failed to process {symbol}: {e}")
            
        time.sleep(0.2) # Prevent Yahoo rate limits

    # 4. Save Watchlist Summary
    with open('public/api/watchlist_summary.json', 'w') as f:
        json.dump(summary_data, f)
        
    print("✅ Static Build Complete!")

if __name__ == "__main__":
    build_static_api()