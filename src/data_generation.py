"""
Generate Realistic Synthetic Oura Ring Data
Based on actual Oura API v2 structure from hedgertronic/oura-ring repo
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data' / 'raw'
DATA_DIR.mkdir(parents=True, exist_ok=True)

class OuraSyntheticDataGenerator:
    """
    Generate realistic Oura Ring data based on:
    1. Actual API structure (from hedgertronic/oura-ring)
    2. Physiological relationships
    3. Individual variability
    4. Temporal patterns
    """
    
    def __init__(self, n_users=50, n_days=90, seed=42):
        np.random.seed(seed)
        self.n_users = n_users
        self.n_days = n_days
        self.start_date = datetime(2024, 1, 1)
        
    def generate_user_baseline(self):
        """Generate baseline characteristics for one user"""
        return {
            'user_id': None,  # Will be set later
            'age': np.random.randint(25, 65),
            'baseline_hr': np.random.normal(55, 8),
            'baseline_hrv': np.random.normal(65, 20),
            'baseline_temp': np.random.normal(0, 0.2),
            'sleep_need': np.random.normal(8.0, 0.5),  # hours
            'activity_level': np.random.choice(['low', 'medium', 'high'], 
                                               p=[0.2, 0.5, 0.3]),
            'chronotype': np.random.choice(['early', 'normal', 'late'],
                                          p=[0.25, 0.50, 0.25])
        }
    
    def calculate_readiness_score(self, sleep_score, activity_yesterday, 
                                   hrv_deviation, temp_deviation, 
                                   sleep_debt, rhr_deviation):
        """
        Reverse-engineered Oura Readiness formula
        Based on their documented contributors
        """
        # Contributor scores (0-100)
        previous_night = sleep_score * 0.35
        
        sleep_balance = max(0, 100 - abs(sleep_debt) * 20)  # Penalize debt
        sleep_balance_contrib = sleep_balance * 0.20
        
        hrv_balance = max(0, 100 - abs(hrv_deviation) * 100)
        hrv_contrib = hrv_balance * 0.15
        
        # Activity balance (moderate is best)
        if 60 <= activity_yesterday <= 85:
            activity_contrib = 100 * 0.15
        else:
            activity_contrib = max(0, 100 - abs(activity_yesterday - 72) * 2) * 0.15
        
        temp_contrib = max(0, 100 - abs(temp_deviation) * 200) * 0.10
        
        rhr_contrib = max(0, 100 - abs(rhr_deviation) * 100) * 0.05
        
        readiness = (previous_night + sleep_balance_contrib + hrv_contrib + 
                    activity_contrib + temp_contrib + rhr_contrib)
        
        return min(100, max(0, readiness))
    
    def generate_daily_sleep(self, user_baseline, day_idx, previous_activity=None):
        """Generate sleep data for one night"""
        
        # Day of week effects
        day_of_week = day_idx % 7
        is_weekend = day_of_week >= 5
        
        # Base sleep duration
        sleep_duration = user_baseline['sleep_need'] * 3600  # seconds
        
        # Weekend adjustment
        if is_weekend:
            sleep_duration += np.random.normal(1800, 600)  # +30 min avg
        
        # Previous activity effect (high activity → more sleep need)
        if previous_activity is not None:
            if previous_activity > 85:
                sleep_duration += np.random.normal(1200, 300)
        
        # Random variation
        sleep_duration += np.random.normal(0, 1800)
        sleep_duration = max(14400, min(36000, sleep_duration))  # 4-10 hours
        
        # Sleep stages (realistic proportions)
        deep_pct = np.clip(np.random.normal(0.18, 0.04), 0.10, 0.25)
        rem_pct = np.clip(np.random.normal(0.22, 0.05), 0.15, 0.30)
        light_pct = 1 - deep_pct - rem_pct - 0.05  # Reserve for awake
        
        deep_duration = sleep_duration * deep_pct
        rem_duration = sleep_duration * rem_pct
        light_duration = sleep_duration * light_pct
        awake_time = sleep_duration * 0.05
        
        # Sleep efficiency
        total_in_bed = sleep_duration + awake_time + np.random.normal(900, 300)
        efficiency = sleep_duration / total_in_bed
        
        # Latency (time to fall asleep)
        latency = max(60, np.random.exponential(600))  # Exponential distribution
        
        # Heart rate during sleep
        hr_avg = user_baseline['baseline_hr'] - np.random.normal(5, 2)
        hr_lowest = hr_avg - np.random.normal(8, 3)
        
        # HRV during sleep
        hrv_avg = user_baseline['baseline_hrv'] + np.random.normal(10, 5)
        
        # Temperature deviation
        temp_dev = user_baseline['baseline_temp'] + np.random.normal(0, 0.15)
        
        # Sleep Score calculation (Oura-like)
        efficiency_score = efficiency * 100
        deep_score = min(100, (deep_duration / 3600) / 1.5 * 100)
        rem_score = min(100, (rem_duration / 3600) / 2.0 * 100)
        latency_score = max(0, 100 - latency / 60 * 10)
        restfulness_score = max(0, 100 - awake_time / 60 * 5)
        
        sleep_score = (efficiency_score * 0.25 + 
                      deep_score * 0.20 + 
                      rem_score * 0.20 + 
                      latency_score * 0.15 + 
                      restfulness_score * 0.20)
        
        return {
            'total_sleep_duration': int(sleep_duration),
            'deep_sleep_duration': int(deep_duration),
            'rem_sleep_duration': int(rem_duration),
            'light_sleep_duration': int(light_duration),
            'awake_time': int(awake_time),
            'sleep_efficiency': round(efficiency, 3),
            'sleep_latency': int(latency),
            'hr_average': round(hr_avg, 1),
            'hr_lowest': round(hr_lowest, 1),
            'hrv_average': round(hrv_avg, 1),
            'temperature_deviation': round(temp_dev, 2),
            'sleep_score': round(sleep_score, 1),
            # Contributors
            'deep_sleep_score': round(deep_score, 1),
            'rem_sleep_score': round(rem_score, 1),
            'efficiency_score': round(efficiency_score, 1),
            'latency_score': round(latency_score, 1),
            'restfulness_score': round(restfulness_score, 1)
        }
    
    def generate_daily_activity(self, user_baseline, day_idx):
        """Generate activity data for one day"""
        
        day_of_week = day_idx % 7
        is_weekend = day_of_week >= 5
        
        # Activity level based on user baseline
        if user_baseline['activity_level'] == 'high':
            base_steps = np.random.normal(12000, 2000)
            high_activity_min = np.random.normal(45, 15)
        elif user_baseline['activity_level'] == 'medium':
            base_steps = np.random.normal(8000, 1500)
            high_activity_min = np.random.normal(25, 10)
        else:
            base_steps = np.random.normal(5000, 1000)
            high_activity_min = np.random.normal(10, 5)
        
        # Weekend adjustment
        if is_weekend:
            base_steps *= np.random.uniform(0.8, 1.3)
            high_activity_min *= np.random.uniform(0.7, 1.4)
        
        steps = max(0, int(base_steps))
        high_activity_time = max(0, int(high_activity_min * 60))  # seconds
        medium_activity_time = int(np.random.normal(3600, 600))
        low_activity_time = int(np.random.normal(7200, 1200))
        
        # Sedentary time (assuming 16 awake hours)
        sedentary_time = 57600 - high_activity_time - medium_activity_time - low_activity_time
        sedentary_time = max(0, sedentary_time)
        
        # Calories (rough estimation)
        # METs: high=8, medium=4, low=2.5, sedentary=1
        user_weight = 70  # kg (assume average)
        calories = (
            user_weight * 8 * (high_activity_time / 3600) +
            user_weight * 4 * (medium_activity_time / 3600) +
            user_weight * 2.5 * (low_activity_time / 3600) +
            user_weight * 1 * (sedentary_time / 3600)
        ) + 1500  # BMR
        
        # Activity Score
        steps_score = min(100, steps / 10000 * 100)
        move_score = min(100, (16 - sedentary_time/3600) / 10 * 100)
        activity_score = (steps_score * 0.4 + move_score * 0.3 + 
                         min(100, high_activity_time/60/30*100) * 0.3)
        
        return {
            'steps': steps,
            'total_calories': int(calories),
            'high_activity_time': high_activity_time,
            'medium_activity_time': medium_activity_time,
            'low_activity_time': low_activity_time,
            'sedentary_time': sedentary_time,
            'activity_score': round(activity_score, 1),
            'steps_score': round(steps_score, 1),
            'move_score': round(move_score, 1)
        }
    
    def generate_dataset(self):
        """Generate complete dataset for all users and days"""
        
        all_data = []
        
        for user_id in range(self.n_users):
            user_baseline = self.generate_user_baseline()
            user_baseline['user_id'] = f'user_{user_id:03d}'
            
            # Track sleep debt
            cumulative_sleep_debt = 0
            
            # Track previous day for readiness calculation
            previous_activity_score = 75
            previous_sleep_score = 75
            
            for day_idx in range(self.n_days):
                current_date = self.start_date + timedelta(days=day_idx)
                
                # Generate sleep
                sleep_data = self.generate_daily_sleep(
                    user_baseline, 
                    day_idx, 
                    previous_activity_score
                )
                
                # Update sleep debt
                sleep_actual = sleep_data['total_sleep_duration'] / 3600
                sleep_debt_today = user_baseline['sleep_need'] - sleep_actual
                cumulative_sleep_debt = 0.5 * cumulative_sleep_debt + 0.5 * sleep_debt_today
                
                # Generate activity
                activity_data = self.generate_daily_activity(user_baseline, day_idx)
                
                # Calculate readiness
                hrv_deviation = (sleep_data['hrv_average'] - user_baseline['baseline_hrv']) / user_baseline['baseline_hrv']
                rhr_deviation = (sleep_data['hr_lowest'] - user_baseline['baseline_hr']) / user_baseline['baseline_hr']
                
                readiness_score = self.calculate_readiness_score(
                    sleep_data['sleep_score'],
                    previous_activity_score,
                    hrv_deviation,
                    sleep_data['temperature_deviation'],
                    cumulative_sleep_debt,
                    rhr_deviation
                )
                
                # Combine all data
                day_data = {
                    'user_id': user_baseline['user_id'],
                    'date': current_date.strftime('%Y-%m-%d'),
                    'day_of_week': day_idx % 7,
                    'is_weekend': int(day_idx % 7 >= 5),
                    
                    # Sleep
                    **sleep_data,
                    
                    # Activity
                    **activity_data,
                    
                    # Readiness
                    'readiness_score': round(readiness_score, 1),
                    'sleep_debt': round(cumulative_sleep_debt, 2),
                    'hrv_deviation': round(hrv_deviation, 3),
                    'rhr_deviation': round(rhr_deviation, 3),
                    
                    # User baseline (for analysis)
                    'user_age': user_baseline['age'],
                    'user_activity_level': user_baseline['activity_level'],
                    'user_chronotype': user_baseline['chronotype']
                }
                
                all_data.append(day_data)
                
                # Update for next day
                previous_activity_score = activity_data['activity_score']
                previous_sleep_score = sleep_data['sleep_score']
        
        df = pd.DataFrame(all_data)
        return df

def main():
    """Main data generation pipeline"""
    print("=" * 60)
    print("GENERATING SYNTHETIC OURA DATA")
    print("=" * 60)
    print("\nBased on actual Oura API v2 structure")
    print("From: hedgertronic/oura-ring GitHub repo")
    print()
    
    # Generate dataset
    print("Generating synthetic Oura data...")
    generator = OuraSyntheticDataGenerator(n_users=50, n_days=90, seed=42)
    df = generator.generate_dataset()
    
    print(f"\n✅ Generated {len(df)} records")
    print(f"   Users: {df['user_id'].nunique()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Save
    output_file = DATA_DIR / 'synthetic_oura_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Data saved to: {output_file}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    summary_cols = ['readiness_score', 'sleep_score', 'activity_score', 
                    'total_sleep_duration', 'steps', 'hrv_average']
    print(df[summary_cols].describe().round(2))
    
    print("\n" + "=" * 60)
    print("✅ DATA GENERATION COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    main()

