"""
Improved LSTM Tuning with Feature Selection and Different Sequence Lengths
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
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False

from train_advanced import create_sequences, TimeSeriesDataset

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'outputs'

class SimpleLSTMModel(nn.Module):
    """Simpler LSTM model - sometimes simpler is better"""
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.2):
        super(SimpleLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last timestep
        out = self.fc(lstm_out[:, -1, :])
        return out, None

def select_time_series_features(df):
    """Select features most relevant for time series prediction"""
    # Core time series features
    core_features = [
        'sleep_score', 'activity_score', 'readiness_score',
        'hrv_average', 'sleep_efficiency', 'steps',
        'sleep_debt', 'temperature_deviation', 'hr_lowest'
    ]
    
    # Lag features (most important)
    lag_features = [col for col in df.columns if 'lag1' in col or 'lag2' in col or 'lag3' in col]
    
    # Rolling averages
    roll_features = [col for col in df.columns if 'roll3d' in col or 'roll7d' in col]
    
    # Training strain
    strain_features = [col for col in df.columns if 'strain' in col or 'load' in col]
    
    # Combine
    selected = core_features + lag_features + roll_features + strain_features
    # Filter to only existing columns
    selected = [f for f in selected if f in df.columns]
    
    return selected

def train_simple_model(X_seq, y_seq, users_seq, feature_size, seq_length,
                       hidden_size=64, num_layers=1, dropout=0.2,
                       learning_rate=0.001, batch_size=32, epochs=50, device='cpu'):
    """Train a simpler LSTM model"""
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_seq, y_seq, users_seq))
    
    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq[train_idx], y_seq[test_idx]
    
    # Scale
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = SimpleLSTMModel(feature_size, hidden_size, num_layers, dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, _ = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            outputs, _ = model(batch_x)
            y_pred.extend(outputs.squeeze().cpu().numpy())
    
    y_pred = np.array(y_pred)
    
    return {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

def main():
    if not HAS_LIBS:
        print("❌ Required libraries not available!")
        return
    
    print("=" * 60)
    print("IMPROVED LSTM TUNING (Feature Selection + Sequence Length)")
    print("=" * 60)
    
    df = pd.read_csv(DATA_DIR / 'forecast_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['user_id', 'date']).reset_index(drop=True)
    
    # Try different approaches
    print("\n1. Testing with selected time series features...")
    selected_features = select_time_series_features(df)
    print(f"   Selected {len(selected_features)} features")
    
    # Test different sequence lengths
    seq_lengths = [3, 5, 7, 10]
    results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for seq_len in seq_lengths:
        print(f"\n2. Testing sequence length: {seq_len} days")
        X_seq, y_seq, users_seq = create_sequences(df, selected_features, seq_length=seq_len)
        print(f"   Sequences: {len(X_seq)}")
        
        result = train_simple_model(
            X_seq, y_seq, users_seq,
            feature_size=len(selected_features),
            seq_length=seq_len,
            hidden_size=64,
            num_layers=1,
            dropout=0.2,
            learning_rate=0.001,
            batch_size=32,
            epochs=50,
            device=device
        )
        
        result['seq_length'] = seq_len
        result['n_features'] = len(selected_features)
        results.append(result)
        
        print(f"   R²: {result['r2']:.3f}, MAE: {result['mae']:.2f}")
    
    # Find best
    best = max(results, key=lambda x: x['r2'])
    
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"Sequence Length: {best['seq_length']} days")
    print(f"Features: {best['n_features']}")
    print(f"R²:  {best['r2']:.3f}")
    print(f"MAE: {best['mae']:.2f} points")
    print(f"RMSE: {best['rmse']:.2f} points")
    
    # Save
    import json
    with open(OUTPUT_DIR / 'improved_tuning_results.json', 'w') as f:
        json.dump({'best': best, 'all': results}, f, indent=2)
    
    print("\n✅ Results saved!")

if __name__ == '__main__':
    main()

