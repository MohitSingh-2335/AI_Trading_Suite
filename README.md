<div align="center">

# 📈 AI Trading Suite

### BTC/USDT Direction Prediction — Research Project

[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-337AB7?style=flat-square)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org/)

**🚧 Active Development — this project is updated regularly as new experiments complete. Code, models, and conclusions may change between visits.**

</div>

---

## 🎯 Motivation

Most public "crypto/stock prediction" projects report strong accuracy but leak future
information into training through random train/test splits. This project asks a narrower,
harder question: **using only a proper walk-forward (no-leakage) validation setup, can any
classical or pretrained model find a directional edge on BTC/USDT hourly candles that
survives real trading fees?**

Rather than stopping at the first model that "looks" profitable, every result here is
stress-tested for statistical significance and replicated across independent time windows
before being trusted.

---

## 📝 What's Been Done So Far

*(Updated after every phase — see linked reports for full methodology)*

| Phase | What was tested | Verdict |
|---|---|---|
| **Phase 0** | Fixed leaky `train_test_split` → chronological split; fixed target framing (`Target_Close` → `Target_Return`); cleaned up config/feature duplication | Foundation corrected — all later results use this |
| **Phase 1** | LogReg / SVC / XGBoost, threshold sweep 0.50–0.70, backtested with real fees | ❌ No replicable edge — best config beat Buy & Hold in only 1 of 5 walk-forward windows (multiple-comparisons false positive) |
| **Phase 2** | Zero-shot Chronos-2, TimesFM 2.5, Moirai 2.0 (pretrained time-series models) | ❌ No exploitable edge — all three ~49% directional accuracy, indistinguishable from chance |
| **Phase 3** | Multi-asset support | ⏳ Not started |
| **Phase 4** | External data agents (Fear & Greed, on-chain, sentiment) | ⏳ Not started |

Full ordered plan tracked internally, phase-by-phase.

---

## 📊 Current Results Summary

- **Best classical model:** Logistic Regression — 52.49% directional accuracy (vs. 49.58%
  naive baseline). Does not survive fee-adjusted backtesting or multi-window validation.
- **Pretrained time-series models (zero-shot):** ~49% directional accuracy — no better
  than chance.
- **Conclusion to date:** No model in this repo has a validated, fee-surviving edge over
  Buy & Hold. This is treated as a real finding, not a setback — see Phase 1/2 reports for
  why each negative result is trustworthy (significance testing + walk-forward replication).

---

## 🔬 Method

- **Data:** BTC/USDT 1-hour OHLCV candles.
- **Features:** 30+ engineered — RSI, MACD, Bollinger Bands, EMA 10/30, rolling
  volume/return stats, lag features, cyclical time encoding.
- **Validation:** Chronological split only, no random shuffling. Any threshold/hyperparameter
  sweep is corrected for multiple comparisons and re-tested on independent time windows
  before being trusted.
- **Backtesting:** Event-driven simulation, 0.1%/side trading fees, Sharpe ratio, max
  drawdown, win rate, equity curve, trade log.
- **App:** Streamlit, 3 modes — Live Prediction (Binance), Simulation (historical
  step-through), Manual Prediction.

---

## 📂 Repository Structure

```
AI_Trading_Suite/
├── app.py
├── config.py
├── prepare_data.py
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── backtesting.py
│   └── agents/                # empty — Phase 4, not built yet
├── scripts/
│   └── sweep_threshold.py
├── models/                    # gitignored, generated locally
├── data/                      # gitignored, generated locally
└── requirements.txt
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/MohitSingh-2335/AI_Trading_Suite.git
cd AI_Trading_Suite

pip install -r requirements.txt

python prepare_data.py          # builds data/featured_btc_data.csv
python -m src.model_training    # trains + saves models/*.pkl

# Add .streamlit/secrets.toml with your own Binance API key/secret
streamlit run app.py
```

---

## 🗺️ Roadmap

Currently deciding between **Phase 3 (multi-asset)** and **Phase 4 (external data
agents)**.

---

## 📄 License
MIT License