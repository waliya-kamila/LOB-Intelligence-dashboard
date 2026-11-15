# LOB Intelligence

### Multi-Horizon Limit Order Book Forecasting Using Deep Learning and Temporal Attention

LOB Intelligence is a research-oriented application for forecasting short-term mid-price movements using deep learning and Limit Order Book (LOB) data.

**Live demo:** https://lob-intelligence-dashboard-b8uf7vy3uc3tw68dshucl5.streamlit.app/

## Overview

The project implements a hybrid CNN-BiGRU architecture with temporal attention to classify future mid-price movements across five forecast horizons.

### Key Features

- Multi-horizon market movement classification
- CNN-BiGRU deep learning architecture
- Temporal attention visualization
- Interactive Streamlit dashboard
- Model performance analysis
- Prediction confidence visualization

## Model Architecture

1. Convolutional Neural Network (CNN)
2. Bidirectional Gated Recurrent Unit (BiGRU)
3. Temporal Attention Mechanism
4. Multi-horizon classification head

## Forecast Configuration

| Parameter | Value |
|---|---|
| Dataset | FI-2010 |
| Dataset subset | NoAuction, anchored folds 1–3 (dashboard model: Fold 1) |
| Input features | 144 |
| Sequence length | 100 |
| Forecast horizons | k = 1, 2, 3, 5, 10 (× 10 order-book events ahead) |
| Prediction classes | UP, STATIONARY, DOWN |

## Key Results

Held-out test macro F1 (mean over the 5 horizons):

| Fold | CNN-BiGRU (no attention) | CNN-BiGRU + Attention |
|---|---|---|
| 1 | 0.6578 | 0.6556 |
| 2 | 0.7526 | 0.7523 |
| 3 | 0.7580 | 0.7537 |
| **Mean ± SD** | **0.7228 ± 0.056** | **0.7205 ± 0.056** |

- Hybrid CNN-BiGRU models clearly beat CNN-only (0.33) and GRU-only (0.50) baselines on Fold 1.
- Temporal attention gives no accuracy gain: 94.8% of attention weight falls on the final time step, and a temporal ablation shows predictions depend almost only on the most recent 25 snapshots.
- Integrated Gradients: spread and mid-price features dominate, with 14 of the top-20 features shared by all three folds.
- Temperature scaling cuts expected calibration error by 54–68% at horizon k = 10.

## Stage 2: Early Detection of Toxic Order Flow

FI-2010 has no trade records, so toxicity is studied on **Binance BTCUSDT perpetual futures**
(Tardis.dev free data, six days in 2024: 1 Mar, 1 Apr, 1 May, 1 Jul, 1 Aug, 1 Oct), which include
a 25-level order book and every trade with its aggressor side.

- **Label:** liquidity providers' markout loss (30-s markouts) on all trades in the next 30 s; the top 20% of
  seconds (threshold from training days) are *toxic*. The model only sees data up to time t, so every
  correct warning comes before the toxic trades happen.
- **Benchmark:** VPIN from exact aggressor-signed volume (bucket = 1/1000 of daily volume, 50-bucket window).
- **Protocol:** walk-forward folds — A: train Mar–May, validate Jul, test Aug; B: train Mar–Jul, validate Aug, test Oct.

Mean test AUC-ROC over the two folds:

| Method | AUC-ROC |
|---|---|
| VPIN | 0.617 |
| Past toxicity (persistence) | 0.577 |
| Logistic regression (11 engineered features) | 0.765 |
| CNN-BiGRU-Attention, order book only | 0.754 |
| **CNN-BiGRU-Attention, order book + trades** | **0.767** |

Integrated Gradients attributes about half of each toxicity warning to the order book's price and depth
structure and half to recent (60-s) volatility and volume. Detailed tables are in `reports/toxicity_results/`.

```bash
python -m scripts.fetch_tardis                              # ~780 MB of raw data into data_crypto/raw
python -m src.toxicity.prepare                              # 1-second arrays per day
TOX_LABEL=exposure python -m src.toxicity.experiment        # main experiment
TOX_LABEL=exposure python -m src.toxicity.explain           # Integrated Gradients
TOX_LABEL=intensity python -m src.toxicity.experiment       # robustness label
```

## Dataset Setup

The project uses the FI-2010 benchmark dataset.

FI-2010 (Ntakaris et al., 2018) is published under CC BY 4.0 on the
[Fairdata repository](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649).

The repository bundles a small demo slice (`deployment_data/demo/fi2010_cf1_test_slice.npz`,
10,100 snapshots of the Fold 1 test day) so the dashboard runs real inference out of the box.
It is regenerated with `python -m scripts.make_demo_slice`.

To use the full Fold 1 test set instead, place the file at:

```text
deployment_data/
└── NoAuction_Zscore_Testing/
    └── Test_Dst_NoAuction_ZScore_CF_1.txt
```

## Model Checkpoint

The dashboard uses the Fold 1 CNN-BiGRU-Attention checkpoint from the multi-fold experiment:

```text
results/checkpoints/deploy_fold1_attention.pt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/waliya-kamila/LOB-Intelligence-dashboard.git
cd LOB-Intelligence-dashboard
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Dashboard

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## Technology Stack

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Streamlit

## Research Scope

This project is designed for research and educational exploration of limit order book forecasting. It uses historical FI-2010 data and does not provide live market predictions or financial advice.