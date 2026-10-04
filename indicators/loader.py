import pandas as pd
import numpy as np

def get_base_dataframe(filepath, tf='D'):
    try:
        # STRICT PARSING to prevent server crashes on dirty CSV files
        df = pd.read_csv(filepath)
        df.columns = [str(c).lower().strip() for c in df.columns] 
        date_col = 'time' if 'time' in df.columns else 'date'
        
        if date_col not in df.columns: 
            return None
            
        df.rename(columns={date_col: 'time'}, inplace=True)
        
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns: 
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else: 
                return None
                
        if 'volume' in df.columns: 
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
        else: 
            df['volume'] = 0.0

        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df.dropna(subset=['time', 'close', 'high', 'low', 'open'], inplace=True)
        
        # --- FILTER OUT YAHOO'S DUMMY HOLIDAY CANDLES ---
        df = df[~((df['volume'] == 0) & (df['high'] == df['low']))]
        
        if df.empty: 
            return None

        # --- SAFELY STRIP TIMEZONES ---
        if df['time'].dt.tz is not None: 
            df['time'] = df['time'].dt.tz_convert(None)
            
        df.drop_duplicates(subset=['time'], keep='last', inplace=True)
        df.sort_values('time', inplace=True)
        
        if len(df) < 2: 
            return None

        # --- TIMEFRAME RESAMPLING (Weekly & Monthly) ---
        if tf == 'W':
            df['grp'] = df['time'].dt.strftime('%G-%V')
            df = df.groupby('grp').agg({'time':'first', 'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
            df.sort_values('time', inplace=True)
        elif tf == 'M':
            df['grp'] = df['time'].dt.strftime('%Y-%m')
            df = df.groupby('grp').agg({'time':'first', 'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}).dropna()
            df.sort_values('time', inplace=True)
            
        df.set_index('time', inplace=True, drop=False)
        return df

    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None