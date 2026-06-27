import numpy as np
import pandas as pd

class ForexRiskEngine:
    def __init__(self, initial_capital=100000.0):
        self.initial_capital = initial_capital

    def calculate_parametric_var(self, returns, confidence=0.95):
        """
        Calculates parametric Value at Risk (VaR) based on variance-covariance properties.
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_dev = np.std(returns)
        
        # Determine standard normal distribution z-score cutoff properties
        # For 95% confidence = 1.645, for 99% confidence = 2.326
        from scipy.stats import norm
        z_score = norm.ppf(confidence)
        
        # Calculate single-day parametric value at risk
        var_pct = -(mean_return - z_score * std_dev)
        return max(0.0, var_pct * self.initial_capital)

    def calculate_conditional_var(self, returns, confidence=0.95):
        """
        Calculates Conditional VaR (Expected Shortfall) capturing average tail-loss severity.
        """
        if len(returns) < 2:
            return 0.0
            
        # Convert absolute returns array into portfolio PnL dollars
        pnl_distribution = returns * self.initial_capital
        
        # Determine parametric VaR cutoff benchmark dollar value
        var_cutoff = self.calculate_parametric_var(returns, confidence)
        
        # Filter distribution entries that violate our peak VaR loss threshold
        tail_losses = pnl_distribution[pnl_distribution <= -var_cutoff]
        
        if len(tail_losses) == 0:
            return var_cutoff
            
        return abs(np.mean(tail_losses))

    def compute_kelly_fraction(self, win_rate, payoff_ratio):
        """
        Applies the Kelly Criterion equation to optimize strategic position sizing.
        Enforces a strict institutional Half-Kelly conservative buffer multiplier.
        """
        if payoff_ratio <= 0:
            return 0.0
            
        # Core standard Kelly allocation formula derivation
        kelly_fraction = win_rate - ((1.0 - win_rate) / payoff_ratio)
        
        # Bound limits to protect against total risk destruction (clip between 0% and 50%)
        kelly_fraction = np.clip(kelly_fraction, 0.0, 0.50)
        
        # Apply strict Half-Kelly protective risk constraints
        half_kelly = kelly_fraction / 2.0
        return half_kelly

if __name__ == "__main__":
    print("=== Initializing Forex Risk Engine Verification ===")
    engine = ForexRiskEngine()
    
    # Extract real backtest results from your Step 2 module execution to verify risk values
    try:
        from backtest import ForexVectorBacktester
        raw_df = pd.read_csv("data/raw/EUR_USD.csv")
        backtester = ForexVectorBacktester()
        results_df = backtester.run_backtest(raw_df)
        
        returns_vector = results_df['Net_Returns'].values
        
        # Calculate dynamic risk thresholds
        var_95 = engine.calculate_parametric_var(returns_vector, 0.95)
        var_99 = engine.calculate_parametric_var(returns_vector, 0.99)
        cvar_95 = engine.calculate_conditional_var(returns_vector, 0.95)
        
        # Gather baseline parameters from our previous backtest metrics run
        win_rate_metric = 0.3011
        profit_factor_metric = 0.9289
        
        fraction = engine.compute_kelly_fraction(win_rate_metric, profit_factor_metric)
        
        print(f"\n=========================================")
        print(f"       RISK SUITE SANITY CHECK LOGS      ")
        print(f"=========================================")
        print(f" 95% Parametric VaR Limit : ${var_95:.2f}")
        print(f" 99% Parametric VaR Limit : ${var_99:.2f}")
        print(f" 95% Conditional VaR (ES) : ${cvar_95:.2f}")
        print(f" Optimized Half-Kelly Sizing: {fraction * 100:.2f}% Allocation")
        print(f"=========================================")
        
    except Exception as e:
        print(f"Pipeline verification failure entry: {str(e)}")