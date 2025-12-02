"""
Train Final Optimized LSTM Model
Using best hyperparameters from tuning
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import pickle
    import json
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

from train_advanced import create_sequences, TimeSeriesDataset

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class OptimizedLSTMModel(nn.Module):
    """Optimized simple LSTM model"""
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(OptimizedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last timestep
        out = self.fc(lstm_out[:, -1, :])
        return out

def select_time_series_features(df):
    """Select features most relevant for time series prediction"""
    core_features = [
        'sleep_score', 'activity_score', 'readiness_score',
        'hrv_average', 'sleep_efficiency', 'steps',
        'sleep_debt', 'temperature_deviation', 'hr_lowest'
    ]
    
    lag_features = [col for col in df.columns if 'lag1' in col or 'lag2' in col or 'lag3' in col]
    roll_features = [col for col in df.columns if 'roll3d' in col or 'roll7d' in col]
    strain_features = [col for col in df.columns if 'strain' in col or 'load' in col]
    
    selected = core_features + lag_features + roll_features + strain_features
    selected = [f for f in selected if f in df.columns]
    
    return selected

def main():
    if not HAS_LIBS:
        print("❌ Required libraries not available!")
        return
    
    print("=" * 60)
    print("TRAINING FINAL OPTIMIZED LSTM MODEL")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_DIR / 'forecast_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['user_id', 'date']).reset_index(drop=True)
    
    # Select features
    print("Selecting time series features...")
    selected_features = select_time_series_features(df)
    print(f"Selected {len(selected_features)} features")
    
    # Create sequences (7 days - best from tuning)
    print("\nCreating sequences (7-day lookback)...")
    X_seq, y_seq, users_seq = create_sequences(df, selected_features, seq_length=7)
    print(f"Sequences: {len(X_seq)}")
    
    # Train/test split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_seq, y_seq, users_seq))
    
    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq[train_idx], y_seq[test_idx]
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model (best config from tuning)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = OptimizedLSTMModel(
        input_size=len(selected_features),
        hidden_size=64,
        num_layers=1,
        dropout=0.2
    ).to(device)
    
    print(f"\nModel architecture:")
    print(f"  Input size: {len(selected_features)}")
    print(f"  Hidden size: 64")
    print(f"  LSTM layers: 1")
    print(f"  Sequence length: 7 days")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/100] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_DIR / 'lstm_optimized_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(MODEL_DIR / 'lstm_optimized_best.pt'))
    
    # Final evaluation
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            y_pred.extend(outputs.squeeze().cpu().numpy())
    
    y_pred = np.array(y_pred)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print("FINAL MODEL RESULTS")
    print(f"{'='*60}")
    print(f"R²:  {r2:.3f}")
    print(f"MAE: {mae:.2f} points")
    print(f"RMSE: {rmse:.2f} points")
    print(f"{'='*60}")
    
    # Save model and scaler
    print("\nSaving model...")
    torch.save(model.state_dict(), MODEL_DIR / 'lstm_optimized_final.pt')
    
    with open(MODEL_DIR / 'lstm_scaler_optimized.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(MODEL_DIR / 'lstm_features_optimized.json', 'w') as f:
        json.dump(selected_features, f, indent=2)
    
    # Save results
    results = {
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'n_features': len(selected_features),
        'seq_length': 7,
        'hidden_size': 64,
        'num_layers': 1
    }
    
    with open(OUTPUT_DIR / 'lstm_optimized_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("✅ Model and results saved!")
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    main()

