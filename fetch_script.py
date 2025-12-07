import yfinance as yf
import pandas as pd
import numpy as np
import random
import time
import requests
import io
from datetime import timedelta, datetime


HISTORY_START_DATE = "2010-01-01"  
NUM_CONTROLS = 10                  
OUTPUT_FILE = "causal_dataset_large.csv"


def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        tables = pd.read_html(io.StringIO(response.text))
        for table in tables:
            if 'Symbol' in table.columns:
                return table['Symbol'].str.replace('.', '-', regex=False).tolist()
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return []
    return []

all_tickers = get_sp500_tickers()
print(f"S&P 500 Tickers Found: {len(all_tickers)}")


print("--- Phase 1: Mining Historical Splits (2010-Present) ---")
print("Scanning all 500 companies... This may take 2-5 minutes.")

found_splits = []


for i, ticker_sym in enumerate(all_tickers):
    try:
        t = yf.Ticker(ticker_sym)
        splits = t.splits
        
        splits = splits[splits.index >= HISTORY_START_DATE]
        


        splits = splits[splits > 1.0]
        
        for date, ratio in splits.items():
            found_splits.append({
                'Ticker': ticker_sym,
                'Split_Date': date
            })
            
    except Exception:
        continue


    if (i+1) % 50 == 0:
        print(f"Scanned {i+1}/{len(all_tickers)} tickers...")

print(f"Total Split Events Found: {len(found_splits)}")
splits_df = pd.DataFrame(found_splits)



def get_historical_features(ticker_symbol, event_date):
    def extract_scalar(value):
        if isinstance(value, (pd.Series, np.ndarray, list)):
            if len(value) > 0:
                return float(value.iloc[0]) if hasattr(value, 'iloc') else float(value[0])
            return 0.0
        return float(value)

    try:
        start_date = event_date - timedelta(days=200)
        end_date = event_date + timedelta(days=10)
        
        # Download data
        df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if df.empty:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        if event_date.tz is not None:
            event_date = event_date.tz_localize(None)

        df = df[df.index <= event_date]
        
        if len(df) < 126: 
            return None

        # 1. Get Current Price
        current_price = extract_scalar(df['Close'].iloc[-1])
        
        # 2. Momentum (6-month)
        price_6m_ago = extract_scalar(df['Close'].iloc[-126])
        momentum_6m = (current_price / price_6m_ago) - 1
            
        # 3. Volatility (30-day)
        returns = df['Close'].pct_change()
        volatility_30d = extract_scalar(returns.tail(30).std())
        
        # 4. Market Cap Proxy (Avg Volume)
        avg_volume_30d = extract_scalar(df['Volume'].tail(30).mean())
        
        return {
            'Ticker': ticker_symbol,
            'Date': event_date,
            'Price': current_price,
            'Momentum_6m': momentum_6m,
            'Volatility_30d': volatility_30d,
            'Avg_Volume_30d': avg_volume_30d
        }

    except Exception as e:
        print(f"  -> Error processing {ticker_symbol}: {e}")
        return None

final_dataset = []

print(f"--- Phase 2: Building Dataset (N={len(splits_df)}) ---")

for index, row in splits_df.iterrows():
    split_ticker = row['Ticker']
    split_date = row['Split_Date']
    
    print(f"[{index+1}/{len(splits_df)}] Processing: {split_ticker} ({split_date.date()})")
    
    # 1. Treated Unit
    features = get_historical_features(split_ticker, split_date)
    if features:
        features['Treated'] = 1
        features['Match_ID'] = index
        final_dataset.append(features)
        
        # 2. Control Units
        potential_controls = [t for t in all_tickers if t != split_ticker]
        
        # Optimization: Sample 15 in case some fail, stop after getting 10 valid ones
        candidates = random.sample(potential_controls, 15)
        controls_collected = 0
        
        for ctrl_ticker in candidates:
            if controls_collected >= NUM_CONTROLS:
                break
                
            ctrl_features = get_historical_features(ctrl_ticker, split_date)
            if ctrl_features:
                ctrl_features['Treated'] = 0
                ctrl_features['Match_ID'] = index
                final_dataset.append(ctrl_features)
                controls_collected += 1
    
    time.sleep(0.5) # Slight delay to be polite to API

if final_dataset:
    causal_df = pd.DataFrame(final_dataset)
    causal_df.dropna(inplace=True)
    
    print("\n--- Collection Complete ---")
    print(f"Total Rows: {len(causal_df)}")
    print(f"Treated (Splits): {len(causal_df[causal_df['Treated']==1])}")
    print(f"Controls (Non-Splits): {len(causal_df[causal_df['Treated']==0])}")
    
    causal_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved to '{OUTPUT_FILE}'")
else:
    print("No data collected.")