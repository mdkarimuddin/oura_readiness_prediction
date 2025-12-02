"""
Evaluate and Visualize Model Results
Compare baseline and LSTM models, visualize attention weights
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
MODEL_DIR = BASE_DIR / 'models'
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model architecture (if needed)
if HAS_TORCH:
    from train_advanced import LSTMAttentionModel, TimeSeriesDataset

def load_baseline_results():
    """Load baseline model results"""
    with open(OUTPUT_DIR / 'baseline_results.json', 'r') as f:
        return json.load(f)

def load_lstm_results():
    """Load LSTM model results"""
    with open(OUTPUT_DIR / 'lstm_results.json', 'r') as f:
        return json.load(f)

def plot_model_comparison(baseline_results, lstm_results):
    """Compare baseline and LSTM models"""
    models = ['Random Forest', 'XGBoost', 'LSTM + Attention']
    r2_scores = [
        baseline_results['Random Forest']['r2'],
        baseline_results['XGBoost']['r2'],
        lstm_results['r2']
    ]
    mae_scores = [
        baseline_results['Random Forest']['mae'],
        baseline_results['XGBoost']['mae'],
        lstm_results['mae']
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² comparison
    axes[0].bar(models, r2_scores, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
    axes[0].set_ylabel('R² Score', fontsize=12)
    axes[0].set_title('Model Comparison: R² Score', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # MAE comparison
    axes[1].bar(models, mae_scores, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.7)
    axes[1].set_ylabel('MAE (points)', fontsize=12)
    axes[1].set_title('Model Comparison: Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    for i, v in enumerate(mae_scores):
        axes[1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: model_comparison.png")

def plot_attention_weights():
    """Visualize attention weights from LSTM model"""
    if not HAS_TORCH:
        print("⚠️  PyTorch not available, skipping attention visualization")
        return
    
    attention_file = OUTPUT_DIR / 'attention_weights.npy'
    if not attention_file.exists():
        print("⚠️  Attention weights file not found")
        return
    
    attention_weights = np.load(attention_file)
    
    # Average attention weights across all samples
    avg_attention = np.mean(attention_weights, axis=0)
    
    # Days in sequence (7 days)
    days = [f'Day -{7-i}' for i in range(7)]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot of average attention
    axes[0].bar(range(7), avg_attention, color='steelblue', alpha=0.7)
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels(days, rotation=45, ha='right')
    axes[0].set_ylabel('Average Attention Weight', fontsize=12)
    axes[0].set_title('Average Attention Weights Across All Predictions', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')
    
    # Add values on bars
    for i, v in enumerate(avg_attention):
        axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
    
    # Heatmap of attention weights (sample of predictions)
    sample_size = min(100, len(attention_weights))
    sample_attention = attention_weights[:sample_size]
    
    im = axes[1].imshow(sample_attention, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    axes[1].set_xlabel('Days in Sequence', fontsize=12)
    axes[1].set_ylabel('Sample Predictions', fontsize=12)
    axes[1].set_title('Attention Weights Heatmap (Sample)', fontsize=14, fontweight='bold')
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(days, rotation=45, ha='right')
    plt.colorbar(im, ax=axes[1], label='Attention Weight')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'attention_weights.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: attention_weights.png")
    
    # Print insights
    print(f"\n📊 Attention Insights:")
    print(f"   Most important day: Day -{7 - np.argmax(avg_attention)} (weight: {np.max(avg_attention):.3f})")
    print(f"   Least important day: Day -{7 - np.argmin(avg_attention)} (weight: {np.min(avg_attention):.3f})")

def plot_predictions_vs_actual():
    """Plot predictions vs actual for LSTM model"""
    # This would require loading the model and making predictions
    # For now, we'll create a placeholder
    print("⚠️  Prediction plots require model inference (to be implemented)")

def main():
    """Main evaluation pipeline"""
    print("=" * 60)
    print("MODEL EVALUATION & VISUALIZATION")
    print("=" * 60)
    
    # Load results
    print("\nLoading model results...")
    baseline_results = load_baseline_results()
    lstm_results = load_lstm_results()
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"\nBaseline Models:")
    for model_name, results in baseline_results.items():
        print(f"  {model_name}:")
        print(f"    R²:  {results['r2']:.3f}")
        print(f"    MAE: {results['mae']:.2f} points")
        print(f"    RMSE: {results['rmse']:.2f} points")
    
    print(f"\nAdvanced Model:")
    print(f"  LSTM + Attention:")
    print(f"    R²:  {lstm_results['r2']:.3f}")
    print(f"    MAE: {lstm_results['mae']:.2f} points")
    print(f"    RMSE: {lstm_results['rmse']:.2f} points")
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    print("\n1. Model comparison...")
    plot_model_comparison(baseline_results, lstm_results)
    
    print("\n2. Attention weights...")
    plot_attention_weights()
    
    print("\n" + "=" * 60)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    main()

