# XDRL for Finance
### Explainable Deep Reinforcement Learning for Portfolio Management

This repository contains the code and data for my project on **explainable post hoc portfolio management with Deep Reinforcement Learning (DRL)**.

The project extends a PPO-based portfolio management framework by integrating **post hoc explainability techniques** to make the agent’s decisions interpretable in trading time:
- **Feature Importance**
- **SHAP**
- **LIME**

The goal is not only to build a DRL agent that allocates portfolio weights, but also to understand **why** it makes each decision.

---

## Publication

This repository supports the open-access paper:

**de-la-Rica-Escudero, A., Garrido-Merchán, E. C., & Coronado-Vaca, M. (2025)**  
*Explainable post hoc portfolio management financial policy of a Deep Reinforcement Learning agent*  
**PLOS ONE, 20(1), e0315528**  
DOI: https://doi.org/10.1371/journal.pone.0315528

---

## Project Overview

### Motivation
Deep Reinforcement Learning has shown strong performance in portfolio management, especially in volatile markets. However, DRL models are often black boxes, which makes them difficult to trust in financial decision-making.

This project addresses that limitation by adding **explainability at prediction time** (not only during training), so users can inspect the agent’s behavior and assess whether its decisions align with an investment policy.

### Main Contribution
- A **PPO-based DRL portfolio management agent**
- A **post hoc explainability layer** for portfolio allocation decisions
- Explanations at both:
  - **Global level** (feature importance)
  - **Local level / per prediction** (SHAP and LIME)

---

## Dataset and Experimental Setup

The experiments use a portfolio of **5 large-cap US technology-related stocks**:
- AAPL (Apple)
- ADBE (Adobe)
- BABA (Alibaba)
- SNE (Sony)
- V (Visa)

### Data sources
- Investing.com
- Wind
- Shinging-Midas Private Fund

### Features
For each asset, the state includes daily:
- **Open**
- **High**
- **Low**
- **Close**

This results in a **20-feature state space** (5 assets × 4 OHLC features), plus the portfolio allocation setting (including cash in the output allocation).

### Time split
- **Training:** 2015-01-01 to 2016-12-31
- **Testing / Trading:** 2017-01-01 to 2018-01-01

---

## Methodology

### 1) DRL Portfolio Allocation (PPO)
The agent is trained with **Proximal Policy Optimization (PPO)** to learn portfolio weights over time by interacting with a financial environment.

### 2) Post hoc Explainability
After training, the project stores state-action pairs and applies explainability methods to interpret the model’s decisions:

- **Feature Importance**  
  Identifies which market inputs are most influential overall.

- **SHAP**  
  Explains the contribution of each feature to a specific portfolio weight prediction.

- **LIME**  
  Provides local, instance-level explanations for the allocation decision at a given time step.

### 3) Explainability Robustness
The project also evaluates the stability of explanations (e.g., Shapley values) across multiple runs and seeds.

---

## Repository Structure

> **Note:** This repository was originally built on top of an existing DRL portfolio management framework and adapted for explainability experiments.

A typical structure is:

- `main.py` — entry point for training, testing, and data workflows
- `agent/` — PPO agent implementation and RL utilities
- `data/` — input datasets (OHLC market data)
- `saved_network/` — trained model checkpoints
- `summary/` — logs, metrics, and experiment summaries
- `environment.py` — portfolio environment and state construction
- `config.json` — experiment configuration
- `explainability/` *(if present in your local version)* — SHAP/LIME/feature importance scripts and outputs
- `results/` *(if present)* — plots and explainability visualizations used in the paper

---

## Installation

### Python version
Recommended:
- **Python 3.9+** (works with modern explainability libraries)

### Core dependencies
Install the main packages (adjust versions to your environment):

```bash
pip install numpy pandas matplotlib scikit-learn
pip install tensorflow
pip install shap lime
