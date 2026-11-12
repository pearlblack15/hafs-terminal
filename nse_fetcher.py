import pandas as pd
from datetime import date, timedelta

def get_nse_delivery(symbol, years=5):
    """
    Fetches historical Delivery Quantity and Delivery Percentage from NSE.
    Returns a Pandas DataFrame with ['Date', 'deliv_vol', 'deliv_pct']
    """
    # NSE api obviously does not support BSE (.BO) tickers. Skip them.
    if symbol.endswith('.BO'):
        return None 
        
    clean_sym = symbol.replace('.NS', '').strip()
    
    try:
        from jugaad_data.nse import stock_df
        to_date = date.today()
        from_date = to_date - timedelta(days=years * 365)
        
        # This single function bypasses Cloudflare and pulls the exact historical data
        df = stock_df(symbol=clean_sym, from_date=from_date, to_date=to_date, series="EQ")
        
        if df.empty:
            return None
            
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Extract only the Date and the two Delivery columns
        res = df[['DATE', 'DELIV_QTY', 'DELIV_PER']].copy()
        res.rename(columns={
            'DATE': 'Date', 
            'DELIV_QTY': 'deliv_vol', 
            'DELIV_PER': 'deliv_pct'
        }, inplace=True)
        
        # Sort chronologically so it cleanly merges with Yahoo Finance data
        res.sort_values('Date', inplace=True)
        return res
        
    except ImportError:
        print("⚠️ 'jugaad-data' not installed. Run: pip install jugaad-data")
        return None
    except Exception as e:
        print(f"⚠️ NSE Delivery Fetch Failed for {clean_sym}: {e}")
        return None
      
