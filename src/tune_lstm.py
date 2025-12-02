"""
Hyperparameter Tuning for LSTM + Attention Model
Grid search to find optimal hyperparameters
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not available")

# Try to import sklearn
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Import model architecture
if HAS_TORCH:
    from train_advanced import LSTMAttentionModel, TimeSeriesDataset, create_sequences

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ImprovedLSTMAttentionModel(nn.Module):
    """
    Improved LSTM model with attention and batch normalization
    """
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(ImprovedLSTMAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers with batch normalization
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        
        # Fully connected layers with batch norm
        self.fc1 = nn.Linear(hidden_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        
        # Attention mechanism
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size)
        
        # Fully connected layers
        out = self.fc1(context)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        
        return out, attention_weights.squeeze(-1)

def train_and_evaluate(X_seq, y_seq, users_seq, feature_size, 
                       hidden_size=128, num_layers=2, dropout=0.3,
                       learning_rate=0.001, batch_size=32, epochs=50,
                       device='cpu', verbose=False):
    """Train and evaluate a single configuration"""
    
    # Train/test split by user
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_seq, y_seq, users_seq))
    
    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq[train_idx], y_seq[test_idx]
    
    # Scale sequences
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_flat)
    
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = ImprovedLSTMAttentionModel(
        input_size=feature_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, _ = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            outputs, _ = model(batch_x)
            y_pred.extend(outputs.squeeze().cpu().numpy())
    
    y_pred = np.array(y_pred)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'best_val_loss': best_val_loss
    }

def hyperparameter_search():
    """Perform grid search over hyperparameters"""
    
    print("=" * 60)
    print("HYPERPARAMETER TUNING FOR LSTM + ATTENTION")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_DIR / 'forecast_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['user_id', 'date']).reset_index(drop=True)
    
    feature_cols = [col for col in df.columns if 
                    col not in ['user_id', 'date', 'target_readiness', 'readiness_score']]
    
    # Create sequences
    print("Creating sequences...")
    X_seq, y_seq, users_seq = create_sequences(df, feature_cols, seq_length=7)
    print(f"Sequences: {len(X_seq)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameter grid
    param_grid = {
        'hidden_size': [64, 128, 256],
        'num_layers': [2, 3],
        'dropout': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.0005],
        'batch_size': [32, 64]
    }
    
    # Reduced grid for faster search (focus on promising combinations)
    configs = [
        {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3, 'learning_rate': 0.001, 'batch_size': 32},
        {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2, 'learning_rate': 0.001, 'batch_size': 32},
        {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.3, 'learning_rate': 0.001, 'batch_size': 32},
        {'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3, 'learning_rate': 0.0005, 'batch_size': 32},
        {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3, 'learning_rate': 0.001, 'batch_size': 64},
        {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.3, 'learning_rate': 0.001, 'batch_size': 32},
    ]
    
    print(f"\nTesting {len(configs)} configurations...")
    print("=" * 60)
    
    results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Testing: {config}")
        
        try:
            result = train_and_evaluate(
                X_seq, y_seq, users_seq,
                feature_size=len(feature_cols),
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                dropout=config['dropout'],
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                epochs=50,
                device=device,
                verbose=False
            )
            
            result.update(config)
            results.append(result)
            
            print(f"  R²: {result['r2']:.3f}, MAE: {result['mae']:.2f}, RMSE: {result['rmse']:.2f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue
    
    # Find best configuration
    if results:
        best_result = max(results, key=lambda x: x['r2'])
        
        print("\n" + "=" * 60)
        print("BEST CONFIGURATION")
        print("=" * 60)
        print(f"R²:  {best_result['r2']:.3f}")
        print(f"MAE: {best_result['mae']:.2f} points")
        print(f"RMSE: {best_result['rmse']:.2f} points")
        print(f"\nHyperparameters:")
        for key, value in best_result.items():
            if key not in ['r2', 'mae', 'rmse', 'best_val_loss']:
                print(f"  {key}: {value}")
        
        # Save results
        import json
        with open(OUTPUT_DIR / 'tuning_results.json', 'w') as f:
            json.dump({
                'best': best_result,
                'all_results': results
            }, f, indent=2)
        
        print(f"\n✅ Results saved to: {OUTPUT_DIR / 'tuning_results.json'}")
        
        return best_result
    else:
        print("\n❌ No successful configurations!")
        return None

def train_best_model(best_config):
    """Train the best model with optimal hyperparameters"""
    print("\n" + "=" * 60)
    print("TRAINING BEST MODEL")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(DATA_DIR / 'forecast_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['user_id', 'date']).reset_index(drop=True)
    
    feature_cols = [col for col in df.columns if 
                    col not in ['user_id', 'date', 'target_readiness', 'readiness_score']]
    
    X_seq, y_seq, users_seq = create_sequences(df, feature_cols, seq_length=7)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train with best config
    result = train_and_evaluate(
        X_seq, y_seq, users_seq,
        feature_size=len(feature_cols),
        hidden_size=best_config['hidden_size'],
        num_layers=best_config['num_layers'],
        dropout=best_config['dropout'],
        learning_rate=best_config['learning_rate'],
        batch_size=best_config['batch_size'],
        epochs=100,  # More epochs for final model
        device=device,
        verbose=True
    )
    
    print(f"\n✅ Final Model Performance:")
    print(f"   R²:  {result['r2']:.3f}")
    print(f"   MAE: {result['mae']:.2f} points")
    print(f"   RMSE: {result['rmse']:.2f} points")
    
    return result

def main():
    """Main tuning pipeline"""
    if not HAS_TORCH or not HAS_SKLEARN:
        print("❌ Required libraries not available!")
        return
    
    # Perform hyperparameter search
    best_config = hyperparameter_search()
    
    if best_config:
        # Train final model with best config
        final_result = train_best_model(best_config)
        
        print("\n" + "=" * 60)
        print("✅ TUNING COMPLETE!")
        print("=" * 60)

if __name__ == '__main__':
    main()

