# Kolmogorov-Arnold Networks with the Multimodal Clean Pareto (KAN-MCP)

This repository hosts the official code for the paper “Towards Explainable Fusion and Balanced Learning in Multimodal Sentiment Analysis”.

## Overview

KAN-MCP, a novel framework that integrates the interpretability of Kolmogorov-Arnold Networks (KAN) with the robustness of the Multimodal Clean Pareto (MCPareto) framework.

![model_architecture.png](attachment:0be3d0a2-0ff4-495b-9adf-a5854f2c71e7:model_architecture.png)

## Datasets and Models Download

### Datasets

```yaml
https://www.dropbox.com/s/sv94igp7zi3rsj1/mosi.pkl?dl=1
https://www.dropbox.com/s/995dvj3506gvk5a/mosei.pkl?dl=1
```

### Models

https://huggingface.co/microsoft/deberta-v3-base

## Quick Start

1. Clone the repository and install dependencies:

```yaml
git clone https://github.com/...
pip install -r requirements.txt
```

1. Download the datasets to `./datasets`
2. Download the Deberta model to `./microsoft/deberta-v3-base`
3. Train the model on MOSI or MOSEI datasets:

```yaml
python train_mosi.py
python train_mosei.py
```
