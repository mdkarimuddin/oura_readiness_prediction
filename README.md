# Forecasting Tomorrow's Recovery: A Deep Learning Approach to Oura Readiness Prediction

Advanced predictive modeling for Oura Ring readiness scores using deep learning with attention mechanisms.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![ML](https://img.shields.io/badge/ML-Deep%20Learning%20%7C%20LSTM-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

## 🎯 Project Overview

**The Innovation**: Most Oura analyses are **descriptive** ("What was my readiness yesterday?")  
This project is **PREDICTIVE** ("What will my readiness be tomorrow?")

### Key Features

1. **Multi-day Forecasting**: Predict readiness 1-7 days ahead
2. **Attention Mechanisms**: Identify which past days matter most for prediction
3. **Uncertainty Quantification**: Provide confidence intervals for predictions
4. **Personalized Calibration**: Learns individual user patterns
5. **Actionable Recommendations**: Suggest what to do today for better tomorrow

## 🔥 What Makes This Outstanding

- ✅ **Uses Oura's Actual API Structure** (studied from GitHub repos)
- ✅ **Synthetic but Realistic Data** (no Oura Ring needed)
- ✅ **Advanced ML** (LSTM + Attention mechanisms)
- ✅ **Production Quality** (proper validation, documentation)
- ✅ **Novel Contribution** (predictive, not descriptive)

## 📊 Dataset

- **Synthetic Oura Data**: Generated based on real Oura API v2 structure
- **Features**: Sleep scores, activity scores, HRV, temperature, readiness contributors
- **Temporal**: Multi-day sequences for time-series prediction
- **Personalization**: User-specific baselines and patterns

## 🗂️ Project Structure

```
oura_readiness_prediction/
├── src/
│   ├── data_generation.py      # Synthetic Oura data generator
│   ├── feature_engineering.py   # Create forecast features
│   ├── train_baseline.py       # Random Forest, XGBoost
│   ├── train_advanced.py       # LSTM with Attention
│   ├── train_final_lstm.py     # Optimized LSTM model
│   ├── train_ensemble.py       # Ensemble model (RF + LSTM)
│   ├── tune_lstm.py            # Hyperparameter tuning
│   └── create_visualizations.py # Model comparison plots
├── data/
│   ├── raw/                    # Generated synthetic data
│   └── processed/             # Processed features
├── models/                     # Trained models
├── outputs/                    # Results & visualizations
├── notebooks/                  # EDA notebooks
└── README.md
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Pipeline

```bash
# Step 1: Generate synthetic Oura data
python src/data_generation.py

# Step 2: Create forecast features
python src/feature_engineering.py

# Step 3: Train baseline models
python src/train_baseline.py

# Step 4: Train advanced LSTM model
python src/train_advanced.py

# Step 4: Train optimized LSTM model
python src/train_final_lstm.py

# Step 5: Train ensemble model
python src/train_ensemble.py

# Step 6: Create visualizations
python src/create_visualizations.py
```

## 🔬 Methodology

### Baseline Models
- **Random Forest Regressor**: Traditional ML approach
- **XGBoost Regressor**: Gradient boosting

### Advanced Model
- **LSTM Optimized**: Deep learning for sequential data
  - 1-layer LSTM architecture (optimized)
  - 64 hidden units
  - 7-day sequence lookback
  - 36 selected time-series features
  - R² = 0.592 (improved from 0.550 through tuning)

### Ensemble Model
- **RF + LSTM Ensemble**: Combines strengths of both models
  - Best configuration: RF 90% + LSTM 10%
  - R² = 0.832, MAE = 2.18 points
  - Provides regularization and robustness

### Features
- Lag features (previous 1, 2, 3, 7 days)
- Rolling averages (3-day, 7-day)
- Training strain (acute:chronic load ratio)
- Sleep debt tracking
- HRV and temperature deviations

## 📈 Results

### Model Performance

| Model | R² | MAE (points) | RMSE (points) | Status |
|-------|----|--------------|---------------|--------|
| **Random Forest** | **0.869** | **1.89** | **2.44** | ✅ Best Model |
| **Ensemble (RF 90% + LSTM 10%)** | 0.832 | 2.18 | 2.77 | ✅ Excellent |
| **XGBoost** | 0.699 | 3.06 | 3.82 | ✅ Strong Baseline |
| **LSTM Optimized** | 0.592 | 3.63 | 4.44 | ✅ Improved from 0.550 |

### Key Achievements

✅ **Random Forest achieves R² = 0.869** - Excellent predictive performance!  
✅ **73 engineered features** created from Oura data structure  
✅ **4,500 synthetic records** (50 users × 90 days)  
✅ **Ensemble model** combining RF + LSTM  
✅ **Hyperparameter tuning** improved LSTM by 7.6%  
✅ **Production-ready** code with proper validation

### Dataset Statistics
- **Total Records**: 4,500 (after feature engineering: 4,100)
- **Users**: 50
- **Time Period**: 90 days per user
- **Features**: 73 engineered features
- **Target**: Tomorrow's readiness score (0-100)

## 💡 Relevance to Oura Ring

This project demonstrates:

✅ **Predictive Health Monitoring** (forecast readiness, not just report it)  
✅ **Deep Learning Expertise** (LSTM, attention mechanisms)  
✅ **Time Series Forecasting** (multi-day ahead predictions)  
✅ **Personalization** (user-specific models)  
✅ **Production-Ready ML** (proper validation, uncertainty quantification)  
✅ **Understanding of Oura Data** (API structure, physiological relationships)

## 🛠️ Technologies

- **Python 3.10+**
- **pandas, numpy** - Data processing
- **scikit-learn** - Baseline ML models
- **XGBoost** - Gradient boosting
- **PyTorch/TensorFlow** - Deep learning (LSTM + Attention)
- **SHAP** - Explainability
- **matplotlib, seaborn** - Visualization

## 👤 Author

**Md Karim Uddin, PhD**  
PhD Veterinary Medicine | MEng Big Data Analytics  
Postdoctoral Researcher, University of Helsinki

- GitHub: [@mdkarimuddin](https://github.com/mdkarimuddin)
- LinkedIn: [Md Karim Uddin, PhD](https://www.linkedin.com/in/md-karim-uddin-phd-aa87649a/)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**⭐ Star this repo if you found it useful!**

*Built to demonstrate advanced ML capabilities for wearable health technology roles.*

