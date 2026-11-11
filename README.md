# LOB Intelligence
### Multi-Horizon Limit Order Book Forecasting Using Deep Learning and Temporal Attention

LOB Intelligence is a research-oriented dashboard for exploring multi-horizon market movement predictions using deep learning and limit order book (LOB) data.

## Overview

The project uses a hybrid CNN-GRU architecture with temporal attention to classify future mid-price movements across five forecast horizons.

### Model Architecture

- Convolutional Neural Network (CNN)
- Bidirectional Gated Recurrent Unit (BiGRU)
- Temporal Attention Mechanism
- Multi-horizon classification head

### Forecast Horizons

1, 2, 3, 5 and 10 steps.

### Prediction Classes

- UP
- STATIONARY
- DOWN

## Dataset

The project uses the FI-2010 benchmark dataset, specifically the NoAuction subset, Fold 1.

- Input features: 144
- Sequence length: 100
- Forecast horizons: 5
- Classification classes: 3

The dataset is not included in this repository. Users must obtain it separately and ensure they comply with its applicable licence and usage terms.

Expected local test-data location:

deployment_data/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_1.txt

## Model Checkpoint

The trained checkpoint is stored at:

results/checkpoints/best_cf1_model.pt

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd TOXIC-ORDER-FLOW

```
