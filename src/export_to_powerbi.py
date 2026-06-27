import os
import pandas as pd
import numpy as np
from backtest import ForexVectorBacktester

def export_star_schema_tables(raw_dir="data/raw", output_dir="data/processed"):
    os.makedirs(output_dir, exist_ok=True)
    backtester = ForexVectorBacktester()
    
    pairs_meta = {
        1: "EUR_USD", 2: "USD_JPY", 3: "GBP_USD", 4: "AUD_USD", 5: "USD_INR"
    }
    
    all_trades = []
    all_market_data = []
    all_signals = []
    
    print("=== Processing Star Schema Data Export ===")
    
    for pair_id, pair_name in pairs_meta.items():
        path = os.path.join(raw_dir, f"{pair_name}.csv")
        if not os.path.exists(path):
            continue
            
        df_raw = pd.read_csv(path, parse_dates=['Date'])
        df_backtest = backtester.run_backtest(df_raw)
        
        # 1. Build FactMarketData Table
        df_market = df_backtest[['Date', 'PairID', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread']].copy()
        df_market['DateTimeKey'] = df_market['Date'].dt.strftime('%Y%m%d')
        all_market_data.append(df_market)
        
        # 2. Build FactSignals Table
        df_sig = df_backtest[['Date', 'PairID', 'Composite_Score', 'Signal']].copy()
        df_sig['SignalID'] = np.arange(len(df_sig)) + (pair_id * 10000)
        df_sig['DateTimeKey'] = df_sig['Date'].dt.strftime('%Y%m%d')
        df_sig['Confidence'] = df_sig['Composite_Score'].abs()
        df_sig['Executed'] = True
        all_signals.append(df_sig)
        
        # 3. Build FactTrades Table from transaction log changes
        trade_rows = df_backtest[df_backtest['Trades'] != 0].copy()
        if not trade_rows.empty:
            trades_df = pd.DataFrame({
                'TradeID': np.arange(len(trade_rows)) + (pair_id * 1000),
                'PairID': pair_id,
                'DateKey': trade_rows['Date'].dt.strftime('%Y%m%d'),
                'StrategyID': 101, # Default Composite Strategy Code
                'Signal': trade_rows['Signal'],
                'PositionSize': 100000,
                'PnL': trade_rows['Net_Returns'] * 100000,
                'IsOpen': False,
                'Direction': np.where(trade_rows['Position'] > 0, "LONG", "SHORT")
            })
            all_trades.append(trades_df)
            
    # Combine and save all Fact Tables
    pd.concat(all_market_data).to_csv(os.path.join(output_dir, "FactMarketData.csv"), index=False)
    pd.concat(all_signals).to_csv(os.path.join(output_dir, "FactSignals.csv"), index=False)
    pd.concat(all_trades).to_csv(os.path.join(output_dir, "FactTrades.csv"), index=False)
    
    # 4. Generate Core Dimension Tables
    dim_currency = pd.DataFrame([
        {"PairID": 1, "PairName": "EUR/USD", "BaseCurrency": "EUR", "QuoteCurrency": "USD", "Region": "G10"},
        {"PairID": 2, "PairName": "USD/JPY", "BaseCurrency": "USD", "QuoteCurrency": "JPY", "Region": "G10"},
        {"PairID": 3, "PairName": "GBP/USD", "BaseCurrency": "GBP", "QuoteCurrency": "USD", "Region": "G10"},
        {"PairID": 4, "PairName": "AUD/USD", "BaseCurrency": "AUD", "QuoteCurrency": "USD", "Region": "Commodity"},
        {"PairID": 5, "PairName": "USD/INR", "BaseCurrency": "USD", "QuoteCurrency": "INR", "Region": "Asian EM"}
    ])
    dim_currency.to_csv(os.path.join(output_dir, "DimCurrencyPair.csv"), index=False)
    
    dim_strategy = pd.DataFrame([
        {"StrategyID": 101, "StrategyName": "Composite_Technical_Voting", "Type": "Quantitative", "RiskLevel": "Medium"}
    ])
    dim_strategy.to_csv(os.path.join(output_dir, "DimStrategy.csv"), index=False)
    
    print("Star Schema dimension and fact matrices successfully written to disk!")

if __name__ == "__main__":
    export_star_schema_tables()