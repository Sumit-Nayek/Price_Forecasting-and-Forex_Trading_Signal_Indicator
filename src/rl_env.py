import gym
from gym import spaces
import numpy as np
import pandas as pd
from ml_models import engineer_ml_features

class ForexTradingEnv(gym.Env):
    """
    A custom modern OpenAI Gym environment for reinforcement learning 
    agent signal generation across structured FX time-series matrices.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=20, initial_balance=100000.0):
        super(ForexTradingEnv, self).__init__()
        
        # Ingest high-dimensional feature matrix
        self.df = engineer_ml_features(df).copy().sort_values('Date').reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        
        # Isolate purely numerical predictive ML features for the observation space
        feature_cols = [col for col in self.df.columns if col not in [
            'Date', 'PairID', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 
            'Signal', 'Target', 'Returns', 'Market_Returns', 'Strategy_Returns', 'Net_Returns'
        ]]
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.close_prices = self.df['Close'].values
        self.spread_costs = self.df['Spread'].values * 0.0001 # Convert pips fraction
        
        # Define Action Space: 0 = Flat/Hold, 1 = Long/Buy, 2 = Short/Sell
        self.action_space = spaces.Discrete(3)
        
        # Define Observation Space: Matrix shape tracking (window_size, total_features)
        num_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.window_size, num_features), 
            dtype=np.float32
        )
        
        self.reset()

    def _get_observation(self):
        """Slices historical frame window up to the active execution step index."""
        return self.features[self.current_step - self.window_size : self.current_step]

    def reset(self, seed=None, options=None):
        """Resets structural accounting loops back to the primary evaluation offset index."""
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.position = 0          # Current holding: 0 = Flat, 1 = Long, -1 = Short
        self.entry_price = 0.0
        self.current_step = self.window_size
        self.total_pnl = 0.0
        
        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        """Processes execution states, evaluating step returns and trade frictions."""
        # Convert explicit discrete action values into portfolio targets
        # action: 0 -> Target position: 0 (Flat)
        # action: 1 -> Target position: 1 (Long)
        # action: 2 -> Target position: -1 (Short)
        target_position = 0
        if action == 1: target_position = 1
        elif action == 2: target_position = -1
        
        current_price = self.close_prices[self.current_step]
        spread = self.spread_costs[self.current_step]
        step_reward = 0.0
        
        # Check if an entry or dynamic position flip occurs
        if target_position != self.position:
            # If currently holding an active side, settle the trade P&L first
            if self.position != 0:
                trade_return = (current_price - self.entry_price) / self.entry_price if self.position == 1 else (self.entry_price - current_price) / self.entry_price
                trade_pnl = trade_return * self.initial_balance
                self.balance += trade_pnl
                self.total_pnl += trade_pnl
                step_reward += trade_pnl
            
            # Apply transactional spread costs logic on execution adjustments
            friction_cost = spread * self.initial_balance
            self.balance -= friction_cost
            step_reward -= friction_cost
            
            # Update systematic execution anchors
            self.position = target_position
            self.entry_price = current_price
            
        else:
            # Passive hold reward generation tracking 
            if self.position != 0:
                prev_price = self.close_prices[self.current_step - 1]
                step_return = (current_price - prev_price) / prev_price if self.position == 1 else (prev_price - current_price) / prev_price
                step_reward += step_return * self.initial_balance
        
        # Increment sequence tracker
        self.current_step += 1
        
        # Check environment boundaries
        terminated = self.current_step >= len(self.df) - 1
        truncated = False # Add for Gym API compatibility compliance
        
        obs = self._get_observation()
        info = {'portfolio_balance': self.balance, 'cumulative_pnl': self.total_pnl}
        
        return obs, step_reward, terminated, truncated, info

if __name__ == "__main__":
    # Internal environment sanity verification script
    raw_df = pd.read_csv("data/raw/EUR_USD.csv")
    env = ForexTradingEnv(raw_df)
    
    obs, info = env.reset()
    print("=== Reinforcement Learning Environment Initialized ===")
    print(f"Observation Matrix Shape: {obs.shape}")
    
    # Run a quick 5-step transaction execution trial loop
    for test_step in range(5):
        random_action = env.action_space.sample()
        next_obs, reward, term, trunc, step_info = env.step(random_action)
        print(f"Step {test_step+1} -> Action Chosen: {random_action} | Step Reward: ${reward:.2f} | Balance: ${step_info['portfolio_balance']:.2f}")