import os
import pandas as pd
import numpy as np

def run_phase_1_eda(raw_dir="data/raw"):
    pairs = ["EUR_USD", "USD_JPY", "GBP_USD", "AUD_USD", "USD_INR"]
    data_dict = {}

    print("=== Starting Codespace Data Quality Suite ===")
    
    # 1. Load and check structural integrity
    for p in pairs:
        path = os.path.join(raw_dir, f"{p}.csv")
        if os.path.exists(path):
            data_dict[p] = pd.read_csv(path, parse_dates=['Date'])
        else:
            print(f"Error: Missing CSV for asset {p} at {path}")
            return

    # 2. Descriptives and Quality Control Report
    print("\n--- ASSET SUMMARY STATS & DESCRIPTIVES ---")
    for name, df in data_dict.items():
        nulls = df.isnull().sum().sum()
        duplicates = df.duplicated(subset=['Date']).sum()
        
        # Calculate dynamic log returns to spot tail behavior/regimes
        df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()
        
        mean_ret = df['Returns'].mean()
        std_ret = df['Returns'].std()
        skew = df['Returns'].skew()
        kurt = df['Returns'].kurtosis() # High kurtosis flags systemic flash crash risks
        
        print(f"\nAsset: {name}")
        print(f"  Row Count: {len(df)} | Missing Elements: {nulls} | Duplicates: {duplicates}")
        print(f"  Log Returns -> Mean: {mean_ret:.6f} | Daily Volatility: {std_ret:.6f}")
        print(f"  Distribution -> Skewness: {skew:.4f} | Excess Kurtosis: {kurt:.4f}")

    # 3. Compute Multi-Pair Cross-Asset Correlation Matrix
    close_prices = {}
    for name, df in data_dict.items():
        close_prices[name] = df.set_index('Date')['Close']

    df_corr = pd.DataFrame(close_prices).pct_change().dropna()
    correlation_matrix = df_corr.corr()

    print("\n--- INSTALMENT MULTI-PAIR CORRELATION MATRIX ---")
    print(correlation_matrix.round(4))
    print("\n=============================================")

if __name__ == "__main__":
    run_phase_1_eda()