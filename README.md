# NBA Total Points Predictor using Deep Learning

This project builds a **neural network regression model** to predict  
the **total combined points scored in an NBA game** using team box score features.  
It uses **PyTorch**, **pandas**, and **scikit-learn** for end-to-end data loading, processing,  
feature engineering, model training, evaluation, and visualization.

---

## Project Overview

### Goal
Predict the **TOTAL_POINTS** (home score + away score) for an NBA game using:
- Shooting efficiency (True Shooting %)
- Game statistics (FGA, REB, AST, TOV)
- Encoded home & away team identities

---

## Dataset

- Input file: `team_traditional.csv`
- Each row: *one team in one game*
- After merging: *one row per game (home + away stats)*

### Key Features Used:
| Feature Type | Features |
|--------------|----------|
| Team Identity | `home_team_enc`, `away_team_enc` |
| Shooting Efficiency | `TS_home`, `TS_away` |
| Offensive Activity | `FGA_home`, `FGA_away` |
| Possession / Control | `REB_home`, `REB_away`, `AST_home`, `AST_away` |
| Mistakes | `TOV_home`, `TOV_away` |
| Target Variables | `home_PTS`, `away_PTS`, `TOTAL_POINTS` |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| ML/Preprocessing | scikit-learn |
| Deep Learning | PyTorch |
| Model Metrics | MAE, MSE, RMSE |

---

## Workflow

1️. Load and filter raw team-level game data  
2️. Merge rows to create **single row per game**  
3️. Engineer **True Shooting % (TS%)** features  
4️. Encode team names using `LabelEncoder`  
5️. Scale both `X` and `y` using `MinMaxScaler`  
6️. Build and train a **3-layer neural network** using PyTorch  
7️. Evaluate using MAE, MSE, RMSE  
8️. Plot training vs validation loss  
9️. Predict real game total points

---

## Model Architecture (PyTorch)

```python
Layer 1: Linear(input_dim → 512) + ReLU
Dropout Layer
Layer 2: Linear(512 → 216) + ReLU
Dropout Layer
Layer 3: Linear(216 → 1)  # Output: Predicted TOTAL_POINTS
Loss: MSELoss
Optimizer: Adam (lr=0.001)
Epochs: 50