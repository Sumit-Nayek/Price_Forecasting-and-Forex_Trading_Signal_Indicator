import os
import pandas as pd
import numpy as np
from indicators import generate_composite_signals

class ForexStressTester:
    def __init__(self, raw_dir="data/raw"):
        self.raw_dir = raw_dir

    def inject_monetary_policy_shock(self, pair_name, shock_type="BOJ_SHOCK"):
        """
        Simulates extreme historical regime transitions directly onto the data matrix
        to stress-test signal engine stability under sudden policy divergence.
        """
        path = os.path.join(self.raw_dir, f"{pair_name}.csv")
        if not os.path.exists(path):
            print(f"Error: Path missing for {pair_name}")
            return None
            
        df = pd.read_csv(path, parse_dates=['Date']).sort_values('Date').reset_index(drop=True)
        
        print(f"\n[STRESS TEST] Injecting '{shock_type}' onto {pair_name} Matrix...")
        
        # Isolate the final 10 days of your dataset to simulate a sudden, concentrated crisis regime
        stress_idx = df.index[-10:]
        
        if shock_type == "BOJ_SHOCK":
            # Simulate a violent carry unwind: asset price gaps down 8% in a compressed window
            # Reminiscent of historical central bank interest rate shocks
            decay_vector = np.linspace(1.0, 0.92, len(stress_idx))
            df.loc[stress_idx, 'Close'] = df.loc[stress_idx, 'Close'] * decay_vector
            df.loc[stress_idx, 'High'] = df.loc[stress_idx, 'High'] * decay_vector
            df.loc[stress_idx, 'Low'] = df.loc[stress_idx, 'Low'] * (decay_vector - 0.01)
            
        elif shock_type == "FLASH_CRASH":
            # Simulate an extreme 15% flash liquidity drain over a multi-day window
            df.loc[stress_idx, 'Close'] = df.loc[stress_idx, 'Close'] * 0.85
            df.loc[stress_idx, 'Spread'] = df.loc[stress_idx, 'Spread'] * 10.0 # Spreads widen 10x
            
        return df

    def evaluate_signal_adaptation(self, stressed_df):
        """
        Evaluates how cleanly the composite engine flips its categorical positions 
        to protect capital when hit by a stressed regime transition.
        """
        # Run the stressed matrix back through your Phase 2 technical consensus engine
        evaluated_df = generate_composite_signals(stressed_df)
        
        # Extract the final 5 steps to verify if the signal flipped to defensive targets
        trailing_window = evaluated_df[['Date', 'Close', 'Spread', 'Composite_Score', 'Signal']].tail(5)
        print("\n--- STRESSED REGIME ENGINE OUTPUT WINDOW ---")
        print(trailing_window.to_string(index=False))
        print("============================================")
        return trailing_window

if __name__ == "__main__":
    tester = ForexStressTester()
    
    # 1. Stress Test USD/JPY against a violent Carry Trade Unwind policy shock
    jpy_stressed = tester.inject_monetary_policy_shock("USD_JPY", "BOJ_SHOCK")
    if jpy_stressed is not None:
        tester.evaluate_signal_adaptation(jpy_stressed)
        
    # 2. Stress Test EUR/USD against an aggressive Flash Crash liquidity drain
    eur_stressed = tester.inject_monetary_policy_shock("EUR_USD", "FLASH_CRASH")
    if eur_stressed is not None:
        tester.evaluate_signal_adaptation(eur_stressed)