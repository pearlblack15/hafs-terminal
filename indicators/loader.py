import pandas as pd
import os

def load_csv(filepath):
    """
    Loads and standardizes historical CSV data for the math engines.
    """
    if not os.path.exists(filepath):
        return pd.DataFrame()
        
    try:
        df = pd.read_csv(filepath)
        
        # 1. Standardize column names
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        if 'date' in df.columns:
            df.rename(columns={'date': 'time'}, inplace=True)
        elif 'datetime' in df.columns:
            df.rename(columns={'datetime': 'time'}, inplace=True)
            
        # 2. Standardize core price columns
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                
        # 3. Handle Volume (The exact source of the error)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
        else:
            df['volume'] = 0.0

        # 4. Handle NSE Delivery Data
        for col in ['deliv_vol', 'deliv_pct']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            else:
                df[col] = 0.0

        # 5. Handle Time formatting and sorting
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            df.dropna(subset=['time'], inplace=True)
            df.sort_values('time', inplace=True)
            
        df.reset_index(drop=True, inplace=True)
        return df

    except Exception as e:
        print(f"❌ Error in loader.py for {filepath}: {e}")
        return pd.DataFrame()

# Fallback alias in case your dispatcher calls it load_data instead of load_csv
def load_data(filepath):
    return load_csv(filepath)
