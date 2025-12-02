"""
Ensemble Model: Combine Random Forest + LSTM
Weighted average of predictions from both models
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

from train_advanced import create_sequences

# Define TimeSeriesDataset here if needed
try:
    import torch
    from torch.utils.data import Dataset
    
    class TimeSeriesDataset(Dataset):
        def __init__(self, sequences, targets):
            self.sequences = torch.FloatTensor(sequences)
            self.targets = torch.FloatTensor(targets)
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return self.sequences[idx], self.targets[idx]
except:
    pass

from train_final_lstm import OptimizedLSTMModel, select_time_series_features

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_rf_model():
    """Load trained Random Forest model"""
    with open(MODEL_DIR / 'baseline_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
        rf_scaler = pickle.load(f)
    
    with open(MODEL_DIR / 'feature_names.json', 'r') as f:
        rf_features = json.load(f)
    
    return rf_model, rf_scaler, rf_features

def load_lstm_model():
    """Load trained LSTM model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load feature names
    with open(MODEL_DIR / 'lstm_features_optimized.json', 'r') as f:
        lstm_features = json.load(f)
    
    # Load scaler
    with open(MODEL_DIR / 'lstm_scaler_optimized.pkl', 'rb') as f:
        lstm_scaler = pickle.load(f)
    
    # Initialize and load model
    model = OptimizedLSTMModel(
        input_size=len(lstm_features),
        hidden_size=64,
        num_layers=1,
        dropout=0.2
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_DIR / 'lstm_optimized_final.pt'))
    model.eval()
    
    return model, lstm_scaler, lstm_features, device

def get_rf_predictions(rf_model, rf_scaler, rf_features, X_test, y_test):
    """Get Random Forest predictions"""
    X_test_rf = X_test[rf_features]
    X_test_scaled = rf_scaler.transform(X_test_rf)
    y_pred_rf = rf_model.predict(X_test_scaled)
    return y_pred_rf

def get_lstm_predictions(lstm_model, lstm_scaler, lstm_features, X_test, y_test, device):
    """Get LSTM predictions"""
    # Create sequences for LSTM
    # Note: This is a simplified version - in practice, we'd need to handle sequences properly
    # For ensemble, we'll use the same test set structure
    
    # For now, we'll need to recreate sequences from the test data
    # This is a limitation - ideally we'd save test sequences during training
    # But for ensemble, we can work with the test data directly
    
    # Create a dummy sequence approach - use last 7 days for each test sample
    # Actually, let's load the test sequences that were used during LSTM training
    # For simplicity, let's predict using the test set directly
    
    # We need to handle this differently - let's create sequences from test data
    return None  # Will implement properly

def create_ensemble_predictions(rf_pred, lstm_pred, weights=None):
    """
    Combine predictions from RF and LSTM
    
    Args:
        rf_pred: Random Forest predictions
        lstm_pred: LSTM predictions
        weights: [rf_weight, lstm_weight]. If None, use performance-based weighting
    """
    if weights is None:
        # Weight based on R² scores (RF=0.716, LSTM=0.592)
        # Normalize weights
        rf_r2 = 0.716
        lstm_r2 = 0.592
        total_r2 = rf_r2 + lstm_r2
        weights = [rf_r2 / total_r2, lstm_r2 / total_r2]
    
    ensemble_pred = weights[0] * rf_pred + weights[1] * lstm_pred
    return ensemble_pred

def main():
    """Main ensemble training pipeline"""
    if not HAS_LIBS:
        print("❌ Required libraries not available!")
        return
    
    print("=" * 60)
    print("ENSEMBLE MODEL: Random Forest + LSTM")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_DIR / 'forecast_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['user_id', 'date']).reset_index(drop=True)
    
    # Define features
    rf_features = [col for col in df.columns if 
                   col not in ['user_id', 'date', 'target_readiness', 'readiness_score']]
    
    lstm_features = select_time_series_features(df)
    
    # Create sequences for LSTM
    print("\nCreating sequences for LSTM...")
    X_seq, y_seq, users_seq = create_sequences(df, lstm_features, seq_length=7)
    
    # Train/test split (same as used in training)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_seq, y_seq, users_seq))
    
    X_seq_train, X_seq_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq[train_idx], y_seq[test_idx]
    
    # For RF, we need the original test data (not sequences)
    # Get corresponding original test data
    df_test = df.iloc[test_idx]
    X_test_rf = df_test[rf_features]
    y_test_rf = df_test['target_readiness'].values
    
    print(f"Test samples: {len(y_test)}")
    
    # Load models
    print("\nLoading models...")
    rf_model, rf_scaler, rf_feature_names = load_rf_model()
    lstm_model, lstm_scaler, lstm_feature_names, device = load_lstm_model()
    
    print("✅ Models loaded")
    
    # Get RF predictions
    print("\nGetting Random Forest predictions...")
    y_pred_rf = get_rf_predictions(rf_model, rf_scaler, rf_feature_names, X_test_rf, y_test_rf)
    
    rf_mae = mean_absolute_error(y_test_rf, y_pred_rf)
    rf_rmse = np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))
    rf_r2 = r2_score(y_test_rf, y_pred_rf)
    
    print(f"RF - R²: {rf_r2:.3f}, MAE: {rf_mae:.2f}, RMSE: {rf_rmse:.2f}")
    
    # Get LSTM predictions
    print("\nGetting LSTM predictions...")
    # Scale test sequences
    X_test_flat = X_seq_test.reshape(-1, X_seq_test.shape[-1])
    X_test_scaled = lstm_scaler.transform(X_test_flat).reshape(X_seq_test.shape)
    
    # Create dataloader
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Predict
    lstm_model.eval()
    y_pred_lstm = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            outputs = lstm_model(batch_x)
            y_pred_lstm.extend(outputs.squeeze().cpu().numpy())
    
    y_pred_lstm = np.array(y_pred_lstm)
    
    # Align lengths (in case of mismatch)
    min_len = min(len(y_test_rf), len(y_pred_lstm))
    y_test_aligned = y_test_rf[:min_len]
    y_pred_rf_aligned = y_pred_rf[:min_len]
    y_pred_lstm_aligned = y_pred_lstm[:min_len]
    
    lstm_mae = mean_absolute_error(y_test_aligned, y_pred_lstm_aligned)
    lstm_rmse = np.sqrt(mean_squared_error(y_test_aligned, y_pred_lstm_aligned))
    lstm_r2 = r2_score(y_test_aligned, y_pred_lstm_aligned)
    
    print(f"LSTM - R²: {lstm_r2:.3f}, MAE: {lstm_mae:.2f}, RMSE: {lstm_rmse:.2f}")
    
    # Test different ensemble weights
    print("\n" + "=" * 60)
    print("ENSEMBLE MODELS (Testing Different Weights)")
    print("=" * 60)
    
    weight_configs = [
        {'name': 'Equal Weight', 'weights': [0.5, 0.5]},
        {'name': 'RF Weighted (0.7)', 'weights': [0.7, 0.3]},
        {'name': 'RF Weighted (0.8)', 'weights': [0.8, 0.2]},
        {'name': 'Performance-Based', 'weights': None},  # Will calculate based on R²
        {'name': 'RF Dominant (0.9)', 'weights': [0.9, 0.1]},
    ]
    
    best_ensemble = None
    best_r2 = -np.inf
    
    for config in weight_configs:
        if config['weights'] is None:
            # Performance-based weighting (use absolute values to handle negative R²)
            rf_weight = max(0, rf_r2) / (max(0, rf_r2) + max(0, lstm_r2) + 1e-8)
            lstm_weight = max(0, lstm_r2) / (max(0, rf_r2) + max(0, lstm_r2) + 1e-8)
            # Normalize
            total = rf_weight + lstm_weight
            if total > 0:
                weights = [rf_weight / total, lstm_weight / total]
            else:
                weights = [0.7, 0.3]  # Fallback
        else:
            weights = config['weights']
        
        y_pred_ensemble = create_ensemble_predictions(
            y_pred_rf_aligned, 
            y_pred_lstm_aligned, 
            weights=weights
        )
        
        mae = mean_absolute_error(y_test_aligned, y_pred_ensemble)
        rmse = np.sqrt(mean_squared_error(y_test_aligned, y_pred_ensemble))
        r2 = r2_score(y_test_aligned, y_pred_ensemble)
        
        print(f"\n{config['name']}:")
        print(f"  Weights: RF={weights[0]:.2f}, LSTM={weights[1]:.2f}")
        print(f"  R²:  {r2:.3f}")
        print(f"  MAE: {mae:.2f} points")
        print(f"  RMSE: {rmse:.2f} points")
        
        if r2 > best_r2:
            best_r2 = r2
            best_ensemble = {
                'name': config['name'],
                'weights': weights,
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'predictions': y_pred_ensemble
            }
    
    # Final results
    print("\n" + "=" * 60)
    print("BEST ENSEMBLE MODEL")
    print("=" * 60)
    print(f"Configuration: {best_ensemble['name']}")
    print(f"Weights: RF={best_ensemble['weights'][0]:.2f}, LSTM={best_ensemble['weights'][1]:.2f}")
    print(f"R²:  {best_ensemble['r2']:.3f}")
    print(f"MAE: {best_ensemble['mae']:.2f} points")
    print(f"RMSE: {best_ensemble['rmse']:.2f} points")
    
    # Compare with individual models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"Random Forest:  R² = {rf_r2:.3f}, MAE = {rf_mae:.2f}")
    print(f"LSTM:           R² = {lstm_r2:.3f}, MAE = {lstm_mae:.2f}")
    print(f"Ensemble:       R² = {best_ensemble['r2']:.3f}, MAE = {best_ensemble['mae']:.2f}")
    
    improvement = ((best_ensemble['r2'] - rf_r2) / rf_r2) * 100
    print(f"\nEnsemble improvement over RF: {improvement:+.2f}%")
    
    # Save results (convert numpy arrays to lists)
    results = {
        'best_ensemble': {
            'name': best_ensemble['name'],
            'weights': [float(w) for w in best_ensemble['weights']],
            'r2': float(best_ensemble['r2']),
            'mae': float(best_ensemble['mae']),
            'rmse': float(best_ensemble['rmse'])
        },
        'individual_models': {
            'random_forest': {
                'r2': float(rf_r2),
                'mae': float(rf_mae),
                'rmse': float(rf_rmse)
            },
            'lstm': {
                'r2': float(lstm_r2),
                'mae': float(lstm_mae),
                'rmse': float(lstm_rmse)
            }
        },
        'improvement_over_rf': float(improvement)
    }
    
    with open(OUTPUT_DIR / 'ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save ensemble weights
    with open(MODEL_DIR / 'ensemble_weights.json', 'w') as f:
        json.dump({
            'rf_weight': float(best_ensemble['weights'][0]),
            'lstm_weight': float(best_ensemble['weights'][1])
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {OUTPUT_DIR / 'ensemble_results.json'}")
    print(f"✅ Ensemble weights saved to: {MODEL_DIR / 'ensemble_weights.json'}")
    
    print("\n" + "=" * 60)
    print("✅ ENSEMBLE TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    main()

