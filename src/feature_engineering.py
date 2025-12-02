"""
Feature Engineering for Readiness Prediction
Create features to predict TOMORROW's readiness from TODAY's data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def create_forecast_features(df):
    """
    Create features to predict TOMORROW's readiness from TODAY's data
    
    Features:
    - Lag features (previous 1, 2, 3, 7 days)
    - Rolling averages (3-day, 7-day)
    - Training strain (acute:chronic load ratio)
    - Sleep debt tracking
    - HRV and temperature deviations
    """
    df_features = df.copy()
    df_features['date'] = pd.to_datetime(df_features['date'])
    df_features = df_features.sort_values(['user_id', 'date']).reset_index(drop=True)
    
    print("Creating forecast features...")
    
    # Lag features (previous days)
    print("  - Lag features (1, 2, 3, 7 days)...")
    for lag in [1, 2, 3, 7]:
        df_features[f'sleep_score_lag{lag}'] = df_features.groupby('user_id')['sleep_score'].shift(lag)
        df_features[f'activity_score_lag{lag}'] = df_features.groupby('user_id')['activity_score'].shift(lag)
        df_features[f'readiness_score_lag{lag}'] = df_features.groupby('user_id')['readiness_score'].shift(lag)
        df_features[f'hrv_avg_lag{lag}'] = df_features.groupby('user_id')['hrv_average'].shift(lag)
        df_features[f'steps_lag{lag}'] = df_features.groupby('user_id')['steps'].shift(lag)
    
    # Rolling averages
    print("  - Rolling averages (3-day, 7-day)...")
    for window in [3, 7]:
        df_features[f'sleep_score_roll{window}d'] = df_features.groupby('user_id')['sleep_score'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df_features[f'activity_score_roll{window}d'] = df_features.groupby('user_id')['activity_score'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df_features[f'readiness_score_roll{window}d'] = df_features.groupby('user_id')['readiness_score'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df_features[f'hrv_avg_roll{window}d'] = df_features.groupby('user_id')['hrv_average'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    
    # Training strain (acute:chronic load ratio)
    print("  - Training strain (acute:chronic)...")
    df_features['acute_load'] = df_features.groupby('user_id')['activity_score'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df_features['chronic_load'] = df_features.groupby('user_id')['activity_score'].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df_features['training_strain'] = df_features['acute_load'] / (df_features['chronic_load'] + 1)
    
    # Sleep debt features
    print("  - Sleep debt features...")
    df_features['sleep_debt_abs'] = np.abs(df_features['sleep_debt'])
    df_features['sleep_debt_roll3d'] = df_features.groupby('user_id')['sleep_debt'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    
    # HRV and temperature trends
    print("  - HRV and temperature trends...")
    df_features['hrv_trend'] = df_features.groupby('user_id')['hrv_average'].transform(
        lambda x: x.rolling(3, min_periods=1).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0)
    )
    df_features['temp_trend'] = df_features.groupby('user_id')['temperature_deviation'].transform(
        lambda x: x.rolling(3, min_periods=1).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) > 1 else 0)
    )
    
    # Current day features (to predict tomorrow)
    print("  - Current day features...")
    current_features = [
        'sleep_score', 'activity_score', 'steps', 'hrv_average',
        'sleep_efficiency', 'deep_sleep_duration', 'rem_sleep_duration',
        'sleep_debt', 'temperature_deviation', 'hr_lowest',
        'day_of_week', 'is_weekend'
    ]
    
    # Target: TOMORROW's readiness
    print("  - Creating target (tomorrow's readiness)...")
    df_features['target_readiness'] = df_features.groupby('user_id')['readiness_score'].shift(-1)
    
    # User characteristics (one-hot encode)
    print("  - User characteristics...")
    df_features = pd.get_dummies(df_features, columns=['user_activity_level', 'user_chronotype'], 
                                  prefix=['activity', 'chronotype'])
    
    print(f"\n✅ Created {len(df_features.columns)} features")
    return df_features

def main():
    """Main feature engineering pipeline"""
    print("=" * 60)
    print("FEATURE ENGINEERING FOR READINESS PREDICTION")
    print("=" * 60)
    
    # Load raw data
    print("\nLoading raw data...")
    df = pd.read_csv(RAW_DIR / 'synthetic_oura_data.csv')
    print(f"Loaded {len(df)} records")
    
    # Create features
    df_features = create_forecast_features(df)
    
    # Remove rows with NaN (from lagging/shifting)
    print("\nRemoving rows with NaN (from lagging/shifting)...")
    initial_len = len(df_features)
    df_features = df_features.dropna()
    print(f"Removed {initial_len - len(df_features)} rows with NaN")
    print(f"Final dataset: {len(df_features)} samples")
    
    # Save processed data
    output_file = PROCESSED_DIR / 'forecast_features.csv'
    df_features.to_csv(output_file, index=False)
    print(f"\n✅ Processed data saved to: {output_file}")
    
    # Feature summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    feature_cols = [col for col in df_features.columns if 
                    col not in ['user_id', 'date', 'target_readiness', 'readiness_score']]
    print(f"Total features: {len(feature_cols)}")
    print(f"\nFeature categories:")
    print(f"  - Lag features: {len([c for c in feature_cols if 'lag' in c])}")
    print(f"  - Rolling averages: {len([c for c in feature_cols if 'roll' in c])}")
    print(f"  - Current features: {len([c for c in feature_cols if c in ['sleep_score', 'activity_score', 'steps', 'hrv_average']])}")
    print(f"  - Derived features: {len([c for c in feature_cols if c in ['training_strain', 'sleep_debt', 'hrv_trend']])}")
    
    print("\n" + "=" * 60)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    main()

