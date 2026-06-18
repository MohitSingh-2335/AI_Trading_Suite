<div align="center">

# 📈 AI Trading Suite

### Multi-Asset ML System — PatchTST Transformer · XGBoost · SVC · Live Data Agents · Backtesting Engine

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit_Cloud-Deployed-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/cloud)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

**Solo Project · Actively Under Development**

> Built a production-ready multi-asset AI trading system using **PatchTST Transformer (2023 SOTA)**, XGBoost, and SVC — with 5-source live data aggregation, proper financial backtesting, and deployed on Streamlit Cloud.

</div>

---

## 🎯 What This Project Is

Most "stock prediction" projects you find online share the same fatal flaw: **random train-test splits on time-series data**. This causes data leakage — the model learns from future data to predict the past, making results look great in notebooks but useless in the real world.

This project is built differently:
- ✅ **Walk-forward validation** — no data leakage, ever
- ✅ **Proper financial evaluation** — Sharpe Ratio, not just accuracy %
- ✅ **Live data pipeline** — 5 independent agents fetch real-time data
- ✅ **2023 state-of-the-art model** — PatchTST Transformer, not LSTM
- ✅ **Deployed production app** — not just a Jupyter notebook

**Supported assets:** BTC · ETH · SOL · AAPL · TSLA · NVDA · and more

---

## 🏗️ System Architecture

```
  ┌─────────────────── 5 Independent Data Agents ───────────────────┐
  │                                                                  │
  │  yfinance API      Binance API      Gemini LLM API              │
  │  (OHLCV data)      (BTC candles)    (news sentiment)            │
  │       │                 │                  │                     │
  │  Fear & Greed      On-Chain                │                     │
  │  Index API         Metrics API             │                     │
  │       │                 │                  │                     │
  └───────┴─────────────────┴──────────────────┴─────────────────── ┘
                             │
                    (merge + clean)
                             │
              ┌──────────────▼─────────────────┐
              │   Feature Engineering (30+)     │
              │   RSI · MACD · Bollinger Bands  │
              │   EMA · Volume · Lag features   │
              │   Cyclical time encoding        │
              └──────────────┬─────────────────┘
                             │
          ┌──────────────────┼─────────────────────┐
          ▼                  ▼                      ▼
  ┌──────────────┐  ┌──────────────────┐  ┌────────────────────┐
  │   XGBoost    │  │       SVC        │  │  PatchTST          │
  │  Regression  │  │  Classification  │  │  Transformer       │
  │              │  │                  │  │  (PyTorch 2023)    │
  │ Predicts:    │  │ Predicts:        │  │                    │
  │ Next hour    │  │ Direction        │  │ Patch-based        │
  │ close price  │  │ UP / DOWN        │  │ time-series        │
  └──────┬───────┘  └──────┬───────────┘  │ forecasting        │
         │                 │              └──────┬─────────────┘
         └─────────────────┴─────────────────────┘
                                    │
                      ┌─────────────▼──────────────┐
                      │     Backtesting Engine      │
                      │  Sharpe · Drawdown · P&L    │
                      │  Equity Curve · Win Rate    │
                      └─────────────┬──────────────┘
                                    │
                      ┌─────────────▼──────────────┐
                      │   Streamlit Cloud App       │
                      │   6-page interactive UI     │
                      │   Live predictions + charts │
                      └────────────────────────────┘
```

---

## 🤖 Models

### PatchTST Transformer *(2023 State-of-the-Art)*
- **Architecture:** Patch-based tokenization → Multi-head self-attention → Forecasting head
- **Why PatchTST over LSTM?** LSTM processes time-series token-by-token. PatchTST divides the series into **patches** (like Vision Transformers do for images), applies attention across patches, and captures long-range dependencies far more effectively. Published in ICLR 2023 and outperforms LSTM on most time-series benchmarks.
- **Training:** GPU-accelerated on NVIDIA RTX 4050 (CUDA 12.1) with early stopping, learning rate scheduling, and model checkpointing
- **Lookback window:** 48 hours

### XGBoost *(Regression)*
- Predicts the **next hour's closing price** (numerical value)
- 30+ engineered features: technical indicators + volume + time cyclical features
- Tuned with `RandomizedSearchCV`

### SVC *(Classification)*
- Predicts **price direction: UP or DOWN** (binary)
- Features normalized with `StandardScaler` inside a `scikit-learn Pipeline`
- Hyperparameter tuning via `RandomizedSearchCV`

---

## 📡 Data Sources — 5 Independent Agents

| Agent | Source | Data |
|-------|--------|------|
| 📊 **Price Agent** | yfinance | Live OHLCV for stocks & crypto |
| 🔶 **Crypto Agent** | Binance API | Real-time BTC/USDT candles |
| 📰 **Sentiment Agent** | Google Gemini API (LLM) | News headline sentiment score |
| 😱 **Fear & Greed Agent** | Alternative.me API | Market fear/greed index (0–100) |
| ⛓️ **On-Chain Agent** | Blockchain.info API | BTC on-chain metrics |

> **Why independent agents?** If the Gemini API goes offline, the other 4 agents still work. The system degrades gracefully — partial data is better than a crash.

---

## 📐 Feature Engineering (30+ Features)

| Category | Features |
|----------|---------|
| **Technical Indicators** | RSI · MACD · Bollinger Bands (high/low) · EMA 10 · EMA 30 |
| **Volume** | Raw volume · volume rolling mean (6h) · volume std (6h) · vol min/max |
| **Price Patterns** | Price change · High-Low ratio · Close-Open diff · Rolling mean/std close |
| **Lag Features** | Previous hour's close · Previous hour's volume |
| **Time Cyclical** | Hour sin/cos · Day-of-week sin/cos (prevents boundary artifacts) |
| **External** | Fear & Greed score · LLM news sentiment score · On-chain transaction volume |

---

## 📊 Backtesting Engine

> **Why backtesting matters:** A model with 55% directional accuracy sounds weak. But with the right strategy, 55% accuracy generates consistent positive returns. Accuracy alone is meaningless — P&L tells the truth.

| Metric | What It Measures |
|--------|-----------------|
| **Sharpe Ratio** | Return per unit of risk (higher = better risk-adjusted returns) |
| **Max Drawdown** | Worst peak-to-trough loss — how much could you lose in the worst case |
| **Win Rate** | % of trades where direction prediction was correct |
| **P&L Simulation** | Full portfolio value simulation: buy on UP signal, sell on DOWN |
| **Equity Curve** | Visual chart of portfolio growth over the backtest period |
| **Trade Log** | Every simulated trade with timestamp, signal, and outcome |

**Validation method:** Walk-forward validation — the model is trained on past data only and tested on future data it has never seen, rolling forward in time. No random splits, no data leakage.

---

## 🖥️ Streamlit App — 6 Pages

| Page | What It Shows |
|------|--------------|
| **Live Prediction** | Real-time data from Binance API → instant model prediction |
| **Simulation** | Step through historical data hour-by-hour, see prediction vs. actual |
| **Manual Input** | Enter your own OHLCV values → get a one-off prediction |
| **Backtesting** | Full strategy backtest with equity curve and trade log |
| **Model Comparison** | XGBoost vs. SVC vs. PatchTST — head-to-head on same data |
| **Analysis** | Feature importance, confusion matrix, error distribution |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Deep Learning** | PyTorch · PatchTST Transformer · CUDA 12.1 · GPU training |
| **Classical ML** | XGBoost · SVC · scikit-learn · RandomizedSearchCV |
| **Data** | pandas · NumPy · yfinance · python-binance · ta (TA-Lib) |
| **AI/LLM** | Google Gemini API (news sentiment analysis) |
| **Visualization** | Streamlit · Plotly (interactive charts) |
| **MLOps** | joblib (model serialization) · conda (environment) · Streamlit Cloud (deploy) |
| **Dev Tools** | Git · GitHub · VS Code · Jupyter Notebook |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/MohitSingh-2335/Stock_Prediction_ML.git
cd Stock_Prediction_ML

# Create and activate environment
conda create -n stock_pred python=3.10
conda activate stock_pred

# Install dependencies
pip install -r requirements.txt

# Add API keys (Binance + Gemini)
# Create .streamlit/secrets.toml (see .env.example)

# Run the app
streamlit run app.py
```

---

## 📂 Project Structure

```
Stock_Prediction_ML/
├── app.py                    # Streamlit multi-page app entry point
├── pages/                    # 6 Streamlit pages
├── src/
│   ├── feature_engineering.py    # 30+ feature pipeline
│   ├── agents/                   # 5 independent data source agents
│   ├── models/                   # PatchTST, XGBoost, SVC training scripts
│   └── backtesting/              # Backtesting engine + metrics
├── models/                   # Serialized model files (.pkl, .pth)
├── data/                     # Cached OHLCV data
├── config.py                 # All constants — no hardcoded values
├── requirements.txt
└── .streamlit/
    ├── config.toml           # Dark theme configuration
    └── secrets.toml          # API keys (gitignored)
```

---

## 📄 License

MIT License
