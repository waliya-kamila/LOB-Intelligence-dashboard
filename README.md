# LOB Intelligence

### Multi-Horizon Limit Order Book Forecasting Using Deep Learning and Temporal Attention

LOB Intelligence is a research-oriented application for forecasting short-term mid-price movements using deep learning and Limit Order Book (LOB) data.

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
| Dataset subset | NoAuction, Fold 1 |
| Input features | 144 |
| Sequence length | 100 |
| Forecast horizons | 1, 2, 3, 5, 10 |
| Prediction classes | UP, STATIONARY, DOWN |

## Dataset Setup

The project uses the FI-2010 benchmark dataset.

The dataset is not included in this repository. Obtain it separately and follow its applicable licence and usage terms.

Place the test file at:

```text
deployment_data/
└── NoAuction_Zscore_Testing/
    └── Test_Dst_NoAuction_ZScore_CF_1.txt
```

## Model Checkpoint

The trained model checkpoint is located at:

```text
results/checkpoints/best_cf1_model.pt
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd TOXIC-ORDER-FLOW
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