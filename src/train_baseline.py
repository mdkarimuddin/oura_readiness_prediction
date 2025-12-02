"""
Train Baseline Models for Readiness Prediction
Random Forest and XGBoost for next-day readiness forecasting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️  XGBoost not available, will only use Random Forest")

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    """Main baseline model training pipeline"""
    print("=" * 60)
    print("BASELINE MODEL TRAINING")
    print("=" * 60)
    
    # Load processed data
    print("\nLoading processed features...")
    df = pd.read_csv(DATA_DIR / 'forecast_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['user_id', 'date']).reset_index(drop=True)
    print(f"Loaded {len(df)} samples")
    
    # Define features and target
    feature_cols = [col for col in df.columns if 
                    col not in ['user_id', 'date', 'target_readiness', 'readiness_score']]
    
    X = df[feature_cols]
    y = df['target_readiness']
    groups = df['user_id']
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Samples: {len(X)}")
    print(f"Target range: {y.min():.1f} - {y.max():.1f}")
    
    # User-based train/test split (important for time series!)
    print("\n" + "=" * 60)
    print("TRAIN/TEST SPLIT (User-based)")
    print("=" * 60)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    train_users = df.iloc[train_idx]['user_id'].unique()
    test_users = df.iloc[test_idx]['user_id'].unique()
    
    print(f"Train users: {len(train_users)} ({len(X_train)} samples)")
    print(f"Test users: {len(test_users)} ({len(X_test)} samples)")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
    }
    
    if HAS_XGBOOST:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
    
    results = {}
    best_score = -np.inf
    best_model = None
    best_name = None
    
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"Test Results:")
        print(f"  MAE:  {mae:.2f} points")
        print(f"  RMSE: {rmse:.2f} points")
        print(f"  R²:   {r2:.3f}")
        
        results[name] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'model': model,
            'predictions': y_pred
        }
        
        if r2 > best_score:
            best_score = r2
            best_model = model
            best_name = name
    
    print("\n" + "=" * 60)
    print(f"✅ BEST MODEL: {best_name}")
    print(f"   R² = {best_score:.3f}")
    print(f"   MAE = {results[best_name]['mae']:.2f} points")
    print("=" * 60)
    
    # Save best model
    print(f"\nSaving best model ({best_name})...")
    with open(MODEL_DIR / 'baseline_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open(MODEL_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(MODEL_DIR / 'feature_names.json', 'w') as f:
        import json
        json.dump(feature_cols, f, indent=2)
    
    print("✅ Models saved!")
    
    # Save results
    results_summary = {name: {k: v for k, v in res.items() if k != 'model' and k != 'predictions'} 
                      for name, res in results.items()}
    
    import json
    with open(OUTPUT_DIR / 'baseline_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✅ Results saved to: {OUTPUT_DIR / 'baseline_results.json'}")
    
    print("\n" + "=" * 60)
    print("✅ BASELINE MODEL TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    main()

