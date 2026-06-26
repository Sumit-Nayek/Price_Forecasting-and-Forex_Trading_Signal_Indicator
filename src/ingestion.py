import os
import pandas as pd
import yfinance as yf

def fetch_and_structure_fx_data(output_dir="data/raw"):
    """
    Fetches historical daily OHLCV data for the 5 project currency pairs,
    standardizes schemas, and writes clean CSV backups to cloud storage.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Mapping project PairIDs to Yahoo Finance tickers [cite: 8, 29]
    fx_pairs = {
        1: {"ticker": "EURUSD=X", "name": "EUR_USD"},
        2: {"ticker": "USDJPY=X", "name": "USD_JPY"},
        3: {"ticker": "GBPUSD=X", "name": "GBP_USD"},
        4: {"ticker": "AUDUSD=X", "name": "AUD_USD"},
        5: {"ticker": "USDINR=X", "name": "USD_INR"}
    }
    
    print("=== Starting Codespace Phase 1 Ingestion Pipeline ===")
    
    for pair_id, meta in fx_pairs.items():
        print(f"Ingesting {meta['name']} (PairID: {pair_id})...")
        
        # Download 2 years of history to provide a sufficient training window
        ticker_data = yf.Ticker(meta["ticker"])
        df = ticker_data.history(period="2y", interval="1d")
        
        if df.empty:
            print(f"Warning: No data found for ticker {meta['ticker']}")
            continue
            
        # Clean index and map variables to match standard schemas
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None) # Strip timezone offsets
        df['PairID'] = pair_id
        
        # Standardize core column nomenclature
        rename_map = {
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        }
        df = df.rename(columns=rename_map)
        
        # Ensure institutional schema integrity
        keep_cols = ['Date', 'PairID', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[keep_cols]
        
        # Set dynamic placeholder corporate spreads based on asset tiers [cite: 47]
        df['Spread'] = 1.2 if pair_id <= 4 else 5.0
        
        # Save output file
        file_path = os.path.join(output_dir, f"{meta['name']}.csv")
        df.to_csv(file_path, index=False)
        print(f"Successfully processed {len(df)} rows. Saved to {file_path}")

if __name__ == "__main__":
    fetch_and_structure_fx_data()