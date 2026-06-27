import os
import pandas as pd
import numpy as np
from indicators import generate_composite_signals

class ForexVectorBacktester:
    def __init__(self, initial_capital=100000.0, default_lot_size=100000.0):
        self.initial_capital = initial_capital
        self.default_lot_size = default_lot_size

    def run_backtest(self, df):
        """
        Executes a historical transaction trace accounting for dynamic spreads, 
        tracking sequential equity curves and trading risk metrics.
        """
        # Ensure signals are cleanly processed
        df = generate_composite_signals(df).copy().sort_values('Date').reset_index(drop=True)
        
        # Numeric parsing for trading positions (1 = Long, -1 = Short, 0 = Flat)
        df['Position'] = 0
        df.loc[df['Signal'].str.contains('BUY'), 'Position'] = 1
        df.loc[df['Signal'].str.contains('SELL'), 'Position'] = -1
        
        # Track position adjustments (Trades occur when configuration flips)
        df['Trades'] = df['Position'].diff().fillna(0)
        
        # Calculate market log-returns
        df['Market_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # System position returns calculation
        df['Strategy_Returns'] = df['Position'].shift(1) * df['Market_Returns']
        
        # FIXED TYPO HERE: Changed pips_unit to pip_unit
        pip_unit = 0.0001
        df['Spread_Cost'] = (df['Trades'].abs() * (df['Spread'] * pip_unit)) / df['Close']
        
        # Net strategy performance after accounting for trading friction
        df['Net_Returns'] = df['Strategy_Returns'] - df['Spread_Cost'].fillna(0)
        
        # Compute equity curves
        df['Cumulative_Market'] = self.initial_capital * np.exp(df['Market_Returns'].cumsum().fillna(0))
        df['Cumulative_Strategy'] = self.initial_capital * np.exp(df['Net_Returns'].cumsum().fillna(0))
        
        # Drawdown calculation vectors
        df['Peak_Equity'] = df['Cumulative_Strategy'].cummax()
        df['Drawdown'] = (df['Cumulative_Strategy'] - df['Peak_Equity']) / df['Peak_Equity']
        
        return df

    def compute_performance_metrics(self, df, name="Asset"):
        """Compiles performance attribution analytics from backtest run matrices."""
        total_trades = int(np.sum(df['Trades'].abs() != 0))
        
        # Approximate wins vs losses via sign changes on trade rows
        trade_rows = df[df['Trades'] != 0].copy()
        trade_rows['Trade_PnL'] = trade_rows['Cumulative_Strategy'].pct_change()
        
        winning_trades = int(np.sum(trade_rows['Trade_PnL'] > 0))
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        gross_profits = trade_rows.loc[trade_rows['Trade_PnL'] > 0, 'Trade_PnL'].sum()
        gross_losses = abs(trade_rows.loc[trade_rows['Trade_PnL'] < 0, 'Trade_PnL'].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
        
        final_equity = df['Cumulative_Strategy'].iloc[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        max_drawdown = df['Drawdown'].min()
        
        # Annualized Sharpe Calculation (Assuming 252 regular trading days matrix)
        daily_mean = df['Net_Returns'].mean()
        daily_std = df['Net_Returns'].std()
        sharpe_ratio = (daily_mean / (daily_std + 1e-10)) * np.sqrt(252) if daily_std > 0 else 0.0

        print(f"\n=========================================")
        print(f" BACKTEST PERFORMANCE METRICS: {name}  ")
        print(f"=========================================")
        print(f" Initial Equity   : ${self.initial_capital:,.2f}")
        print(f" Final Net Equity : ${final_equity:,.2f}")
        print(f" Total Return     : {total_return * 100:.2f}%")
        print(f" Total Trades Run : {total_trades}")
        print(f" Win Rate Metric  : {win_rate * 100:.2f}%")
        print(f" Profit Factor    : {profit_factor:.4f}")
        print(f" Max Drawdown Peak: {max_drawdown * 100:.2f}%")
        print(f" Annualized Sharpe: {sharpe_ratio:.4f}")
        print(f"=========================================")
        
        return {
            "Asset": name, "Total_Return": total_return, "Trades": total_trades,
            "Win_Rate": win_rate, "Profit_Factor": profit_factor, 
            "Max_DD": max_drawdown, "Sharpe": sharpe_ratio
        }

if __name__ == "__main__":
    # Validate framework using raw EUR_USD dataset
    raw_df = pd.read_csv("data/raw/EUR_USD.csv")
    backtester = ForexVectorBacktester()
    results_df = backtester.run_backtest(raw_df)
    backtester.compute_performance_metrics(results_df, name="EUR_USD")