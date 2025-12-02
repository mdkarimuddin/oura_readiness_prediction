"""
Train Advanced LSTM Model with Attention Mechanism
Deep learning approach for next-day readiness prediction
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
    print("⚠️  PyTorch not available, cannot train LSTM model")
    print("   Please load pytorch module: module load pytorch/2.7")

# Try to import sklearn for scaling
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️  sklearn not available")

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LSTM with Attention Model
if HAS_TORCH:
    class AttentionLayer(nn.Module):
        """
        Attention mechanism to weight important time steps
        """
        def __init__(self, hidden_size):
            super(AttentionLayer, self).__init__()
            self.hidden_size = hidden_size
            self.W = nn.Linear(hidden_size, hidden_size)
            self.v = nn.Linear(hidden_size, 1, bias=False)
            
        def forward(self, lstm_output):
            # lstm_output shape: (batch, seq_len, hidden_size)
            # Compute attention scores
            attention_scores = self.v(torch.tanh(self.W(lstm_output)))  # (batch, seq_len, 1)
            attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
            
            # Weighted sum
            context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch, hidden_size)
            
            return context, attention_weights.squeeze(-1)  # (batch, hidden_size), (batch, seq_len)
    
    class LSTMAttentionModel(nn.Module):
        """
        LSTM model with attention for readiness prediction
        """
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
            super(LSTMAttentionModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            # LSTM layers
            self.lstm = nn.LSTM(
                input_size, 
                hidden_size, 
                num_layers, 
                batch_first=True, 
                dropout=dropout if num_layers > 1 else 0
            )
            
            # Attention layer
            self.attention = AttentionLayer(hidden_size)
            
            # Fully connected layers
            self.fc1 = nn.Linear(hidden_size, 32)
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(32, 1)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            # x shape: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
            
            # Apply attention
            context, attention_weights = self.attention(lstm_out)  # (batch, hidden_size), (batch, seq_len)
            
            # Fully connected layers
            out = self.fc1(context)
            out = self.relu(out)
            out = self.dropout(out)
            out = self.fc2(out)
            
            return out, attention_weights
    
    class TimeSeriesDataset(Dataset):
        """Dataset for time series sequences"""
        def __init__(self, sequences, targets):
            self.sequences = torch.FloatTensor(sequences)
            self.targets = torch.FloatTensor(targets)
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return self.sequences[idx], self.targets[idx]

def create_sequences(df, feature_cols, seq_length=7):
    """
    Create sequences for time-series prediction
    Each sequence: past 7 days → predict tomorrow
    """
    sequences = []
    targets = []
    user_ids = []
    
    for user in df['user_id'].unique():
        user_data = df[df['user_id'] == user].sort_values('date').reset_index(drop=True)
        
        if len(user_data) < seq_length + 1:
            continue
        
        features = user_data[feature_cols].values
        target = user_data['target_readiness'].values
        
        for i in range(len(features) - seq_length):
            # Skip if target is NaN
            if pd.isna(target[i + seq_length]):
                continue
                
            sequences.append(features[i:i+seq_length])
            targets.append(target[i + seq_length])
            user_ids.append(user)
    
    return np.array(sequences), np.array(targets), np.array(user_ids)

def train_lstm_model(X_seq, y_seq, users_seq, feature_size, epochs=50, batch_size=32, 
                     learning_rate=0.001, device='cpu'):
    """Train LSTM with Attention model"""
    
    # Train/test split by user
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_seq, y_seq, users_seq))
    
    X_train, X_test = X_seq[train_idx], X_seq[test_idx]
    y_train, y_test = y_seq[train_idx], y_seq[test_idx]
    
    print(f"Train sequences: {len(X_train)}")
    print(f"Test sequences: {len(X_test)}")
    
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
    model = LSTMAttentionModel(
        input_size=feature_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"\nModel architecture:")
    print(f"  Input size: {feature_size}")
    print(f"  Hidden size: 64")
    print(f"  LSTM layers: 2")
    print(f"  Sequence length: 7 days")
    print(f"  Training on: {device}")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print(f"\n{'='*60}")
    print("TRAINING LSTM + ATTENTION MODEL")
    print(f"{'='*60}")
    
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
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), MODEL_DIR / 'lstm_attention_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(MODEL_DIR / 'lstm_attention_best.pt'))
    
    # Final evaluation
    model.eval()
    y_pred = []
    attention_weights_all = []
    
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            outputs, attention_weights = model(batch_x)
            y_pred.extend(outputs.squeeze().cpu().numpy())
            attention_weights_all.extend(attention_weights.cpu().numpy())
    
    y_pred = np.array(y_pred)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print("LSTM + ATTENTION RESULTS")
    print(f"{'='*60}")
    print(f"MAE:  {mae:.2f} points")
    print(f"RMSE: {rmse:.2f} points")
    print(f"R²:   {r2:.3f}")
    print(f"{'='*60}")
    
    # Save scaler and attention weights
    import pickle
    with open(MODEL_DIR / 'lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save attention weights for visualization
    attention_weights_all = np.array(attention_weights_all)
    np.save(OUTPUT_DIR / 'attention_weights.npy', attention_weights_all)
    
    return model, scaler, {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred,
        'actual': y_test,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

def main():
    """Main advanced model training pipeline"""
    print("=" * 60)
    print("ADVANCED LSTM + ATTENTION MODEL TRAINING")
    print("=" * 60)
    
    if not HAS_TORCH:
        print("\n❌ PyTorch not available!")
        print("   Please load: module load pytorch/2.7")
        return
    
    if not HAS_SKLEARN:
        print("\n❌ sklearn not available!")
        return
    
    # Load processed data
    print("\nLoading processed features...")
    df = pd.read_csv(DATA_DIR / 'forecast_features.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['user_id', 'date']).reset_index(drop=True)
    print(f"Loaded {len(df)} samples")
    
    # Define features
    feature_cols = [col for col in df.columns if 
                    col not in ['user_id', 'date', 'target_readiness', 'readiness_score']]
    
    print(f"\nFeatures: {len(feature_cols)}")
    
    # Create sequences
    print("\nCreating sequences (7-day lookback)...")
    seq_length = 7
    X_seq, y_seq, users_seq = create_sequences(df, feature_cols, seq_length)
    
    print(f"Sequences created: {len(X_seq)}")
    print(f"Sequence shape: (samples, time_steps, features) = {X_seq.shape}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Train model
    model, scaler, results = train_lstm_model(
        X_seq, y_seq, users_seq,
        feature_size=len(feature_cols),
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        device=device
    )
    
    # Save model
    print(f"\nSaving model...")
    torch.save(model.state_dict(), MODEL_DIR / 'lstm_attention_model.pt')
    print("✅ Model saved!")
    
    # Save results
    import json
    results_summary = {
        'mae': float(results['mae']),
        'rmse': float(results['rmse']),
        'r2': float(results['r2'])
    }
    
    with open(OUTPUT_DIR / 'lstm_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"✅ Results saved to: {OUTPUT_DIR / 'lstm_results.json'}")
    
    print("\n" + "=" * 60)
    print("✅ ADVANCED MODEL TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    main()

