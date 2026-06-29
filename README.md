# Price Forecasting & Forex Trading Signal Indicator

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/ML-Time%20Series-orange?style=for-the-badge&logo=tensorflow" />
</p>

<p align="center">
  A machine learning–powered system for forex price forecasting and automated buy/sell signal generation, combining deep learning models with technical analysis indicators.
</p>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Technical Indicators](#technical-indicators)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

---

## 🔍 Overview

This project addresses the challenge of **forex price prediction** and **automated signal generation** using a hybrid approach of deep learning–based time series forecasting and classical technical analysis. The system ingests historical OHLCV (Open, High, Low, Close, Volume) data for currency pairs, trains predictive models, and overlays trading signals to assist in decision-making.

The indicator pipeline processes raw market data and outputs:
- **Short-term price forecasts** (next N candles)
- **Buy / Sell / Hold signals** derived from predicted trends and indicator crossovers
- **Risk metrics** including confidence intervals and volatility estimates

> ⚠️ **Disclaimer:** This project is developed for academic and research purposes. It is not financial advice. Live trading involves significant risk of loss.

---

## ✨ Features

- 📊 **Multi-currency support** — EUR/USD, GBP/USD, USD/JPY, and more
- 🧠 **Deep learning forecasting** — LSTM / GRU / Transformer-based price prediction
- 📉 **Technical indicator engine** — RSI, MACD, Bollinger Bands, EMA, SMA, ATR, Stochastic Oscillator
- 🚦 **Signal generation** — Rule-based + ML-driven BUY / SELL / HOLD signals
- 📅 **Multi-timeframe analysis** — Support for M15, H1, H4, D1 timeframes
- 📈 **Backtesting module** — Evaluate strategy performance on historical data
- 🔌 **Modular pipeline** — Easily swap models, indicators, or data sources
- 📁 **Visual output** — Candlestick charts annotated with signals and forecasts

---

## 🏗️ Architecture

```
Raw OHLCV Data (CSV / API)
        │
        ▼
┌───────────────────┐
│  Data Preprocessing│  ← Normalization, Cleaning, Feature Engineering
└────────┬──────────┘
         │
         ▼
┌────────────────────────────────┐
│  Feature Extraction            │
│  ├── Technical Indicators       │
│  │    (RSI, MACD, BB, EMA ...)  │
│  └── Lag Features / Windows     │
└────────┬───────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌─────────────────┐
│ ML     │ │ Deep Learning   │
│ Models │ │ (LSTM / GRU /   │
│(XGBoost│ │  Transformer)   │
│ RF ...) │ └────────┬────────┘
└────┬───┘          │
     └──────┬────────┘
            ▼
   ┌────────────────┐
   │ Signal Engine  │  ← Merge predictions + indicator crossovers
   │  BUY/SELL/HOLD │
   └────────┬───────┘
            ▼
   ┌────────────────┐
   │  Backtesting   │  ← P&L, Win Rate, Sharpe Ratio, Max Drawdown
   └────────┬───────┘
            ▼
   ┌────────────────┐
   │  Visualization │  ← Annotated charts, performance plots
   └────────────────┘
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3.8+ |
| **Deep Learning** | TensorFlow / Keras, PyTorch |
| **ML** | Scikit-learn, XGBoost |
| **Data Processing** | Pandas, NumPy |
| **Technical Analysis** | `ta`, `pandas-ta`, or custom implementations |
| **Visualization** | Matplotlib, Plotly, mplfinance |
| **Backtesting** | Custom engine / Backtrader |
| **Data Source** | Yahoo Finance (`yfinance`), Alpha Vantage, or CSV |

---

## 📂 Dataset

Historical OHLCV data for major forex pairs is used for training and evaluation.

| Field | Description |
|---|---|
| `Open` | Opening price of the candle |
| `High` | Highest price in the period |
| `Low` | Lowest price in the period |
| `Close` | Closing price of the candle |
| `Volume` | Trade volume (tick volume for forex) |

**Sources:**
- [Yahoo Finance](https://finance.yahoo.com/) via `yfinance`
- [Alpha Vantage](https://www.alphavantage.co/) API
- Historical CSV exports from MetaTrader 4/5

**Timeframes available:** M1, M5, M15, H1, H4, D1

> ℹ️ Place raw data files in `data/raw/` before running the pipeline.

---

## 🧠 Models Used

### 1. LSTM (Long Short-Term Memory)
Captures long-range temporal dependencies in sequential price data. The model is trained on sliding windows of OHLCV + indicator values to predict next-step close price.

### 2. GRU (Gated Recurrent Unit)
A lighter alternative to LSTM with similar performance on shorter sequences, trained in parallel for comparison.

### 3. Transformer / Attention-Based Model *(optional)*
Self-attention mechanism to weight the relevance of different time steps in the input sequence.

### 4. Ensemble / Hybrid
Final signal combines deep learning price forecast (directional bias) with rule-based indicator signals to produce robust BUY / SELL / HOLD decisions.

---

## 📐 Technical Indicators

| Indicator | Purpose |
|---|---|
| **EMA (9, 21, 50, 200)** | Trend direction |
| **SMA** | Baseline moving average |
| **MACD** | Momentum and crossover signals |
| **RSI (14)** | Overbought / oversold detection |
| **Bollinger Bands** | Volatility bands and breakout detection |
| **ATR (14)** | Volatility measurement |
| **Stochastic Oscillator** | Momentum / reversal signals |
| **ADX** | Trend strength |

All indicators are computed from raw OHLCV data inside `src/features/indicators.py`.

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip / conda

### Clone the repository

```bash
git clone https://github.com/Sumit-Nayek/Price_Forecasting-and-Forex_Trading_Signal_Indicator.git
cd Price_Forecasting-and-Forex_Trading_Signal_Indicator
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda create -n forex-env python=3.10
conda activate forex-env
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Download or prepare data

```bash
python src/data/download_data.py --pair EURUSD --timeframe H1 --start 2018-01-01 --end 2024-01-01
```

### 2. Generate features

```bash
python src/features/build_features.py --input data/raw/EURUSD_H1.csv --output data/processed/
```

### 3. Train the forecasting model

```bash
python src/models/train.py --model lstm --pair EURUSD --epochs 100 --seq_len 60
```

### 4. Generate trading signals

```bash
python src/signals/generate_signals.py --pair EURUSD --model_path models/lstm_EURUSD.h5
```

### 5. Backtest the strategy

```bash
python src/backtest/run_backtest.py --pair EURUSD --signals data/signals/EURUSD_signals.csv
```

### 6. Visualize results

```bash
python src/visualization/plot_signals.py --pair EURUSD
```

---

## 📊 Results

> *Note: Update this section with your actual experimental results.*

| Model | Currency Pair | MAE | RMSE | Direction Accuracy |
|---|---|---|---|---|
| LSTM | EUR/USD | — | — | —% |
| GRU | EUR/USD | — | — | —% |
| Ensemble | EUR/USD | — | — | —% |

**Backtest Summary (EUR/USD, H1, 2022–2024):**

| Metric | Value |
|---|---|
| Total Trades | — |
| Win Rate | —% |
| Profit Factor | — |
| Sharpe Ratio | — |
| Max Drawdown | —% |

---

## 📁 Project Structure

```
Price_Forecasting-and-Forex_Trading_Signal_Indicator/
│
├── data/
│   ├── raw/                  # Raw OHLCV CSV files
│   └── processed/            # Feature-engineered datasets
│
├── models/                   # Saved model weights (.h5 / .pt)
│
├── notebooks/                # Jupyter notebooks for EDA and experiments
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Signal_Generation.ipynb
│
├── src/
│   ├── data/
│   │   └── download_data.py   # Data ingestion from yfinance / Alpha Vantage
│   ├── features/
│   │   └── indicators.py      # Technical indicator computations
│   │   └── build_features.py  # Feature pipeline
│   ├── models/
│   │   ├── lstm_model.py      # LSTM architecture
│   │   ├── gru_model.py       # GRU architecture
│   │   └── train.py           # Training entry point
│   ├── signals/
│   │   └── generate_signals.py # Signal generation logic
│   ├── backtest/
│   │   └── run_backtest.py    # Backtesting engine
│   └── visualization/
│       └── plot_signals.py    # Chart rendering
│
├── requirements.txt
├── config.yaml                # Hyperparameters and paths
└── README.md
```

---

## 🔮 Future Work

- [ ] Real-time signal generation with live API feed
- [ ] Reinforcement learning agent for adaptive position sizing
- [ ] Sentiment analysis integration (news + social data)
- [ ] MetaTrader 5 Expert Advisor integration for automated execution
- [ ] Web dashboard for live signal monitoring (Streamlit / FastAPI)
- [ ] Multi-asset portfolio optimization module

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please ensure code follows PEP 8 and includes appropriate docstrings.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Sumit Nayek**
Project Research Scientist | AI/ML Researcher
NITTTR Kolkata

- GitHub: [@Sumit-Nayek](https://github.com/Sumit-Nayek)
- LinkedIn: [linkedin.com/in/sumit-nayek](https://linkedin.com/in/sumit-nayek) *(update link)*

---

<p align="center">
  Made with ❤️ for research and algorithmic trading exploration
</p>
