"""
Create Comprehensive Visualizations
Model comparison, predictions, and performance metrics
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

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = BASE_DIR / 'outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_results():
    """Load all model results"""
    baseline_results = {}
    lstm_results = {}
    ensemble_results = {}
    
    try:
        with open(OUTPUT_DIR / 'baseline_results.json', 'r') as f:
            baseline_results = json.load(f)
    except:
        pass
    
    try:
        with open(OUTPUT_DIR / 'lstm_optimized_results.json', 'r') as f:
            lstm_results = json.load(f)
    except:
        pass
    
    try:
        with open(OUTPUT_DIR / 'ensemble_results.json', 'r') as f:
            ensemble_results = json.load(f)
    except:
        pass
    
    return baseline_results, lstm_results, ensemble_results

def plot_model_comparison(baseline_results, lstm_results, ensemble_results):
    """Compare all models"""
    models = []
    r2_scores = []
    mae_scores = []
    colors = []
    
    # Add baseline models
    if 'Random Forest' in baseline_results:
        models.append('Random Forest')
        r2_scores.append(baseline_results['Random Forest']['r2'])
        mae_scores.append(baseline_results['Random Forest']['mae'])
        colors.append('#3498db')
    
    if 'XGBoost' in baseline_results:
        models.append('XGBoost')
        r2_scores.append(baseline_results['XGBoost']['r2'])
        mae_scores.append(baseline_results['XGBoost']['mae'])
        colors.append('#2ecc71')
    
    # Add LSTM
    if lstm_results and 'r2' in lstm_results:
        models.append('LSTM Optimized')
        r2_scores.append(lstm_results['r2'])
        mae_scores.append(lstm_results['mae'])
        colors.append('#e74c3c')
    
    # Add ensemble
    if ensemble_results and 'best_ensemble' in ensemble_results:
        models.append('Ensemble\n(RF 90% + LSTM 10%)')
        r2_scores.append(ensemble_results['best_ensemble']['r2'])
        mae_scores.append(ensemble_results['best_ensemble']['mae'])
        colors.append('#9b59b6')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # R² comparison
    bars1 = axes[0].bar(models, r2_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('R² Score', fontsize=13, fontweight='bold')
    axes[0].set_title('Model Comparison: R² Score', fontsize=15, fontweight='bold', pad=20)
    axes[0].set_ylim([0, 1])
    axes[0].grid(alpha=0.3, axis='y')
    axes[0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
    axes[0].axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Excellent (0.8)')
    axes[0].legend(loc='lower right')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # MAE comparison
    bars2 = axes[1].bar(models, mae_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('MAE (points)', fontsize=13, fontweight='bold')
    axes[1].set_title('Model Comparison: Mean Absolute Error', fontsize=15, fontweight='bold', pad=20)
    axes[1].grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars2, mae_scores)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: model_comparison.png")

def plot_performance_metrics(baseline_results, lstm_results, ensemble_results):
    """Create comprehensive performance metrics plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    models = []
    r2_scores = []
    mae_scores = []
    rmse_scores = []
    
    # Collect data
    if 'Random Forest' in baseline_results:
        models.append('RF')
        r2_scores.append(baseline_results['Random Forest']['r2'])
        mae_scores.append(baseline_results['Random Forest']['mae'])
        rmse_scores.append(baseline_results['Random Forest']['rmse'])
    
    if 'XGBoost' in baseline_results:
        models.append('XGB')
        r2_scores.append(baseline_results['XGBoost']['r2'])
        mae_scores.append(baseline_results['XGBoost']['mae'])
        rmse_scores.append(baseline_results['XGBoost']['rmse'])
    
    if lstm_results and 'r2' in lstm_results:
        models.append('LSTM')
        r2_scores.append(lstm_results['r2'])
        mae_scores.append(lstm_results['mae'])
        rmse_scores.append(lstm_results['rmse'])
    
    if ensemble_results and 'best_ensemble' in ensemble_results:
        models.append('Ensemble')
        r2_scores.append(ensemble_results['best_ensemble']['r2'])
        mae_scores.append(ensemble_results['best_ensemble']['mae'])
        rmse_scores.append(ensemble_results['best_ensemble']['rmse'])
    
    x_pos = np.arange(len(models))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(models)]
    
    # R² Score
    axes[0, 0].bar(x_pos, r2_scores, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(models, fontsize=12)
    axes[0, 0].set_ylabel('R² Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('R² Score Comparison', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(r2_scores):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    
    # MAE
    axes[0, 1].bar(x_pos, mae_scores, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(models, fontsize=12)
    axes[0, 1].set_ylabel('MAE (points)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Mean Absolute Error', fontsize=13, fontweight='bold')
    axes[0, 1].grid(alpha=0.3, axis='y')
    for i, v in enumerate(mae_scores):
        axes[0, 1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
    
    # RMSE
    axes[1, 0].bar(x_pos, rmse_scores, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models, fontsize=12)
    axes[1, 0].set_ylabel('RMSE (points)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Root Mean Squared Error', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(rmse_scores):
        axes[1, 0].text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold')
    
    # Combined metrics (normalized)
    r2_norm = [r / max(r2_scores) for r in r2_scores]
    mae_norm = [1 - (m / max(mae_scores)) for m in mae_scores]  # Invert so higher is better
    rmse_norm = [1 - (r / max(rmse_scores)) for r in rmse_scores]
    
    x = np.arange(len(models))
    width = 0.25
    
    axes[1, 1].bar(x - width, r2_norm, width, label='R² (normalized)', alpha=0.7)
    axes[1, 1].bar(x, mae_norm, width, label='MAE (inverted)', alpha=0.7)
    axes[1, 1].bar(x + width, rmse_norm, width, label='RMSE (inverted)', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, fontsize=12)
    axes[1, 1].set_ylabel('Normalized Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Combined Metrics (Normalized)', fontsize=13, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: performance_metrics.png")

def plot_ensemble_weights(ensemble_results):
    """Visualize ensemble weight configurations"""
    if not ensemble_results or 'best_ensemble' not in ensemble_results:
        print("⚠️  Ensemble results not available")
        return
    
    # This would require storing all weight configurations
    # For now, just show the best one
    best = ensemble_results['best_ensemble']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    weights = best['weights']
    labels = ['Random Forest', 'LSTM']
    colors = ['#3498db', '#e74c3c']
    explode = (0.05, 0) if weights[0] > weights[1] else (0, 0.05)
    
    wedges, texts, autotexts = ax.pie(weights, labels=labels, colors=colors, 
                                       autopct='%1.1f%%', startangle=90,
                                       explode=explode, shadow=True)
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    ax.set_title(f'Best Ensemble Configuration\n{best["name"]}\nR² = {best["r2"]:.3f}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'ensemble_weights.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: ensemble_weights.png")

def main():
    """Main visualization pipeline"""
    print("=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    # Load results
    print("\nLoading model results...")
    baseline_results, lstm_results, ensemble_results = load_results()
    
    # Create visualizations
    print("\n1. Model comparison...")
    plot_model_comparison(baseline_results, lstm_results, ensemble_results)
    
    print("\n2. Performance metrics...")
    plot_performance_metrics(baseline_results, lstm_results, ensemble_results)
    
    print("\n3. Ensemble weights...")
    plot_ensemble_weights(ensemble_results)
    
    print("\n" + "=" * 60)
    print("✅ VISUALIZATIONS COMPLETE!")
    print("=" * 60)
    print(f"\nAll plots saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()

