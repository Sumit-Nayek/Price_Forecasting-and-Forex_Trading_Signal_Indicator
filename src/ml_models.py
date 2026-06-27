import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from indicators import generate_composite_signals

def engineer_ml_features(df):
    """
    Transforms basic technical indicator matrices into a high-dimensional 
    feature array tailored for machine learning prediction.
    """
    # 1. Calculate the missing log returns vector before indicators consume it
    df = df.copy().sort_values('Date').reset_index(drop=True)
    df['Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 2. Apply baseline composite and indicator engineering modules
    df = generate_composite_signals(df).copy()
    
    # 3. Momentum & Return Lags
    for lag in [1, 3, 5, 10]:
        df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)
        df[f'Vol_Lag_{lag}'] = df['Returns'].rolling(window=lag+2).std()
    
    # 4. Structural Interaction Boundaries
    df['Spread_to_ATR'] = df['Spread'] / (df['ATR_14'] + 1e-10)
    df['Distance_BB_Upper'] = df['BB_Upper'] - df['Close']
    df['Distance_BB_Lower'] = df['Close'] - df['BB_Lower']
    df['MACD_Slope'] = df['MACD_Hist'].diff(2)
    
    # 5. Define Binary Target Vector (1 if next-day price shifts up, else 0)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df.dropna()

def train_and_evaluate_classifiers(df, asset_name="EUR_USD"):
    """
    Executes an out-of-sample walk-forward cross-validation strategy 
    across Random Forest and XGBoost classifiers.
    """
    df_ml = engineer_ml_features(df)
    
    # Exclude non-predictive features from feature matrix X
    feature_cols = [col for col in df_ml.columns if col not in [
        'Date', 'PairID', 'Open', 'High', 'Low', 'Close', 'Volume', 'Spread', 
        'Signal', 'Target', 'Returns', 'Market_Returns', 'Strategy_Returns', 'Net_Returns'
    ]]
    
    X = df_ml[feature_cols]
    y = df_ml['Target']
    
    # Enforce chronological sequence integrity to eliminate data leakage
    tscv = TimeSeriesSplit(n_splits=5)
    
    rf_acc, xgb_acc = [], []
    
    print(f"\n=========================================")
    print(f" TRAINING ML CLASSIFIERS: {asset_name}   ")
    print(f"=========================================")
    print(f" Total Samples Extracted: {len(df_ml)}")
    print(f" Features Active Counter: {len(feature_cols)}")
    print(f"=========================================")
    
    # Cross-validation iteration loop
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Initialize Models
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
        xgb_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=42)
        
        # Fit Models
        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)
        
        # Predict Validation Sequences
        rf_preds = rf_model.predict(X_test)
        xgb_preds = xgb_model.predict(X_test)
        
        # Record Model Accuracy
        fold_rf_acc = accuracy_score(y_test, rf_preds)
        fold_xgb_acc = accuracy_score(y_test, xgb_preds)
        rf_acc.append(fold_rf_acc)
        xgb_acc.append(fold_xgb_acc)
        
        print(f" Fold {fold+1} Validation -> RF Acc: {fold_rf_acc*100:.2f}% | XGB Acc: {fold_xgb_acc*100:.2f}%")
        
    print(f"-----------------------------------------")
    print(f" Mean RF Cross-Val Accuracy : {np.mean(rf_acc)*100:.2f}%")
    print(f" Mean XGB Cross-Val Accuracy: {np.mean(xgb_acc)*100:.2f}%")
    print(f"=========================================\n")
    
    # Print a classification report for the final testing slice
    print("XGBOOST FINAL FOLD DETAILED PERFORMANCE REPORT:")
    print(classification_report(y_test, xgb_preds))

if __name__ == "__main__":
    raw_df = pd.read_csv("data/raw/EUR_USD.csv")
    train_and_evaluate_classifiers(raw_df, asset_name="EUR_USD")