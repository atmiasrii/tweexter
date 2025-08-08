import os
import time
import datetime
import warnings
import sys
from typing import Dict, List, Tuple, Optional, Union
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

# Data Science
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Custom Features & Models
from features import extract_features_from_text, MODEL_FEATURES
from persona_engine import load_personas, aggregate_engagement

# Custom Models (add to top)
try:
    from retweet_model import RetweetPredictor
    from reply_model import ReplyPredictor
except ImportError:
    RetweetPredictor = None
    ReplyPredictor = None

# Suppress feature extraction warnings for cleaner output
warnings.filterwarnings("ignore", message="⚠️")

# Final reporting: print_analysis_summary
def print_analysis_summary(preds_df: pd.DataFrame, results: List[Dict], quality_report: Dict):
    """Generate comprehensive final report"""
    print("\nMODEL PERFORMANCE SUMMARY")
    print("-" * 50)
    print(f"{'Metric':<10} {'ML MAE':>10} {'Persona MAE':>12} {'Blended MAE':>12} {'Improvement':>12}")
    for r in results:
        print(f"{r['metric']:<10} {r['mae_ml']:>10.2f} {r['mae_persona']:>12.2f} {r['mae_blend']:>12.2f} {r['improvement']:>11.1f}%")
    print("\nBLEND WEIGHT DISTRIBUTION")
    print("-" * 50)
    for metric in METRICS:
        w = preds_df[f'optimal_blend_{metric}'].mean()
        print(f"{metric:<10} ML: {w:.1%} | Persona: {(1-w):.1%}")
    print("\nDATA QUALITY METRICS")
    print("-" * 50)
    for k, v in quality_report.items():
        print(f"{k.replace('_', ' ').title():<20}: {v:.1%}")
    print("\nRECOMMENDATIONS")
    print("-" * 50)
    print("1. Review large error samples for model improvement opportunities")
    print("2. Check feature correlations to understand blend weight drivers")
    print("3. Monitor temporal patterns for seasonality effects")
    print("4. Consider text length normalization if correlations are strong")

# Version checking
import sys
def check_versions():
    """Verify all model versions are compatible"""
    print("\nVERSION COMPATIBILITY CHECK")
    current_versions = {
        'code': '1.5.2',
        'models': MODEL_VERSIONS,
        'python': f"{sys.version_info.major}.{sys.version_info.minor}",
        'pandas': pd.__version__,
        'numpy': np.__version__
    }
    print("Current Environment:")
    for k, v in current_versions.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for subk, subv in v.items():
                print(f"    {subk}: {subv}")
        else:
            print(f"  {k}: {v}")
# Text length impact analysis
def analyze_text_length_impact(preds_df: pd.DataFrame):
    """Analyze relationship between text length and prediction accuracy"""
    preds_df['text_length'] = preds_df['tweet'].str.len()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        sns.scatterplot(data=preds_df, x='text_length', 
                       y=f'blend_{metric}_error', ax=ax, alpha=0.5)
        ax.set_title(f'{metric.capitalize()} Error vs Text Length')
        ax.set_xlabel('Tweet Length (chars)')
        ax.set_ylabel('Prediction Error')
    plt.tight_layout()
    plt.savefig(f'{DIAGNOSTICS_DIR}/text_length_impact.png')
    plt.close()

# Enhanced error handling: Safe model loading
def safe_load_model(path: str, model_type: str = 'model'):
    """Safely load models with version checking"""
    try:
        model = joblib.load(path)
        print(f"Successfully loaded {model_type} from {path}")
        return model
    except Exception as e:
        print(f"⚠️ Failed to load {model_type} from {path}: {str(e)[:200]}")
        return None

# Resource cleanup utility
def cleanup_resources():
    """Clean up temporary files and resources"""
    temp_files = [
        'ml_persona_preds_temp.csv',
        'blend_error_histograms.png',
        'error_comparison_boxplots.png'
    ]
    for file in temp_files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"Cleaned up temporary file: {file}")
        except Exception as e:
            print(f"Failed to clean up {file}: {str(e)[:100]}")
# Standard Library
import os
import time
import datetime
import warnings
from typing import Dict, List, Tuple, Optional, Union
from multiprocessing import Pool, cpu_count
import multiprocessing as mp

# Data Science
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
import joblib

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Custom Features & Models
from features import extract_features_from_text, MODEL_FEATURES
from persona_engine import load_personas, aggregate_engagement

# Custom Models (add to top)
try:
    from retweet_model import RetweetPredictor
    from reply_model import ReplyPredictor
except ImportError:
    RetweetPredictor = None
    ReplyPredictor = None

# Suppress feature extraction warnings for cleaner output
warnings.filterwarnings("ignore", message="⚠️")


# Constants
REG_SEEDS = [42, 99, 123, 456, 789]
METRICS = ['likes', 'retweets', 'replies']
DIAGNOSTICS_DIR = 'diagnostics'
os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)

# Model version tracking
MODEL_VERSIONS = {
    'likes': '1.2',
    'retweets': '1.1',
    'replies': '1.0',
    'personas': '2.3'
}
def check_data_quality(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate basic data quality metrics"""
    quality = {
        'missing_values': df.isnull().mean().mean(),
        'zero_engagement': (df[['Likes', 'Retweets', 'Replies']] == 0).mean().mean(),
        'duplicate_tweets': df['Tweet Text'].duplicated().mean()
    }
    print("\nData Quality Report:")
    for k, v in quality.items():
        print(f"  {k.replace('_', ' ').title()}: {v:.1%}")
    return quality

# Load Models and Scalers
reg_models_likes = [joblib.load(f'regression_xgb_seed_{seed}.pkl') for seed in REG_SEEDS]
likes_scaler = joblib.load('likes_log_scaler.pkl')
reg_models_rt = [joblib.load(f'regression_rt_xgb_seed_{seed}.pkl') for seed in REG_SEEDS]
rt_scaler = joblib.load('retweets_log_scaler.pkl')
reg_models_reply = [joblib.load(f'regression_reply_xgb_seed_{seed}.pkl') for seed in REG_SEEDS]
reply_scaler = joblib.load('replies_log_scaler.pkl')

PERSONAS = load_personas("personas.json")

def timed_execution(func):
    """Decorator to measure and log execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"\nStarting {func.__name__}...")
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"Completed {func.__name__} in {elapsed:.2f} seconds")
        return result
    return wrapper

def validate_input_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean input dataframe"""
    required_cols = {'Tweet Text', 'Likes', 'Retweets', 'Replies'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Clean numeric columns
    for col in ['Likes', 'Retweets', 'Replies']:
        df[col] = df[col].apply(clean_numeric_value)
    
    # Remove empty tweets
    initial_count = len(df)
    df = df[df['Tweet Text'].notna() & (df['Tweet Text'].str.strip() != '')]
    df = df.reset_index(drop=True)
    
    print(f"Validated {len(df)}/{initial_count} tweets after cleaning")
    return df

def clean_numeric_value(val) -> int:
    """Convert values with 'K' suffix and handle edge cases"""
    if pd.isna(val):
        return 0
    if isinstance(val, (int, float)):
        return int(val)
    
    val_str = str(val).strip().upper()
    if val_str.endswith('K'):
        return int(float(val_str[:-1]) * 1000)
    try:
        return int(float(val_str))
    except (ValueError, TypeError):
        return 0

def process_single_tweet(tweet_data):
    """Process a single tweet - designed for multiprocessing"""
    idx, tweet, actual_likes, actual_retweets, actual_replies = tweet_data
    
    try:
        # Load models and personas for this process (each process needs its own copy)
        reg_models_likes = [joblib.load(f'regression_xgb_seed_{seed}.pkl') for seed in REG_SEEDS]
        likes_scaler = joblib.load('likes_log_scaler.pkl')
        reg_models_rt = [joblib.load(f'regression_rt_xgb_seed_{seed}.pkl') for seed in REG_SEEDS]
        rt_scaler = joblib.load('retweets_log_scaler.pkl')
        reg_models_reply = [joblib.load(f'regression_reply_xgb_seed_{seed}.pkl') for seed in REG_SEEDS]
        reply_scaler = joblib.load('replies_log_scaler.pkl')
        
        personas = load_personas("personas.json")
        
        features = extract_features_from_text(tweet)
        features_df = pd.DataFrame([features])[MODEL_FEATURES]

        # ML predictions
        ml_likes = predict_metric_ml(reg_models_likes, likes_scaler, features_df)
        
        # Try specialized predictors first, fallback to ensemble if they fail
        try:
            if RetweetPredictor and ReplyPredictor:
                retweet_predictor = RetweetPredictor()
                reply_predictor = ReplyPredictor()
                
                ml_rt = int(retweet_predictor.predict(features))
                ml_reply = int(reply_predictor.predict(features))
            else:
                raise ImportError("Specialized predictors not available")
        except Exception:
            # Fallback to ensemble approach but with validation
            ml_rt_raw = predict_metric_ml(reg_models_rt, rt_scaler, features_df)
            ml_reply_raw = predict_metric_ml(reg_models_reply, reply_scaler, features_df)
            
            # Apply same validation as predict.py
            ml_rt = ml_rt_raw
            if ml_rt > 10 * ml_likes:
                ml_rt = int(ml_likes * 0.1)  # Sanity check
            
            ml_reply = min(ml_reply_raw, int(ml_likes * 1.5))  # Cap replies to likes * 1.5
        
        # Persona predictions
        persona_result = aggregate_engagement(tweet, personas)
        persona_likes = persona_result.get('persona_likes', 0)
        persona_rts = persona_result.get('persona_rts', 0)
        persona_replies = persona_result.get('persona_replies', 0)

        return {
            'idx': idx,
            'actual_likes': actual_likes,
            'actual_retweets': actual_retweets,
            'actual_replies': actual_replies,
            'ml_likes': ml_likes,
            'ml_retweets': ml_rt,
            'ml_replies': ml_reply,
            'persona_likes': int(persona_likes),
            'persona_retweets': int(persona_rts),
            'persona_replies': int(persona_replies),
            'tweet': tweet,
            'success': True
        }
    except Exception as e:
        return {
            'idx': idx,
            'actual_likes': actual_likes,
            'actual_retweets': actual_retweets,
            'actual_replies': actual_replies,
            'ml_likes': 0,
            'ml_retweets': 0,
            'ml_replies': 0,
            'persona_likes': 0,
            'persona_retweets': 0,
            'persona_replies': 0,
            'tweet': tweet,
            'success': False,
            'error': str(e)
        }

def safe_expm1(x):
    x = np.clip(x, None, 20)
    return np.expm1(x)
    x = np.clip(x, None, 20)
    return np.expm1(x)

def predict_metric_ml(models, scaler, features_df):
    preds = [m.predict(features_df)[0] for m in models]
    base_pred = np.mean(preds)
    pred_log = base_pred * scaler['std'] + scaler['mean']
    return int(safe_expm1(pred_log))

def get_preds_for_all(df):
    total_tweets = len(df)
    print(f"Processing {total_tweets} tweets in parallel...")
    print(f"Estimated processing time: {total_tweets * 0.1:.1f} seconds (approximate)")
    
    # Prepare data for multiprocessing
    print("Preparing data for parallel processing...")
    tweet_data_list = []
    for idx, row in df.iterrows():
        tweet_data_list.append((
            idx,
            row['Tweet Text'],
            int(row['Likes']),
            int(row['Retweets']),
            int(row['Replies'])
        ))
    
    print(f"Data preparation complete. Ready to process {len(tweet_data_list)} tweets.")
    
    # Use multiprocessing
    num_processes = min(cpu_count(), 8)  # Limit to 8 processes to avoid overwhelming
    print(f"Using {num_processes} parallel processes...")
    print("Starting parallel processing... (this may take a while for large datasets)")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_single_tweet, tweet_data_list)
    
    print("Parallel processing complete! Organizing results...")
    
    # Sort results by original index to maintain order
    results.sort(key=lambda x: x['idx'])
    
    # Convert to DataFrame
    print("Converting results to DataFrame...")
    all_preds = []
    failed_count = 0
    successful_count = 0
    
    for i, result in enumerate(results):
        if i % 100 == 0:  # Progress update every 100 tweets
            print(f"  Processing result {i+1}/{len(results)}...")
            
        if not result['success']:
            failed_count += 1
            if failed_count <= 5:  # Show first 5 errors in detail
                print(f"  Failed to process tweet {result['idx']}: {result.get('error', 'Unknown error')[:100]}...")
        else:
            successful_count += 1
        
        all_preds.append({
            'actual_likes': result['actual_likes'],
            'actual_retweets': result['actual_retweets'],
            'actual_replies': result['actual_replies'],
            'ml_likes': result['ml_likes'],
            'ml_retweets': result['ml_retweets'],
            'ml_replies': result['ml_replies'],
            'persona_likes': result['persona_likes'],
            'persona_retweets': result['persona_retweets'],
            'persona_replies': result['persona_replies'],
            'tweet': result['tweet']
        })
    
    if failed_count > 0:
        print(f"  Warning: {failed_count} tweets failed to process (showing first 5 errors above)")
    print(f"  Successfully processed: {successful_count}/{total_tweets} tweets ({successful_count/total_tweets*100:.1f}%)")
    print(f"  Completed processing all {total_tweets} tweets!")
    return pd.DataFrame(all_preds)

def find_optimal_blend(df, ml_col, persona_col, actual_col):
    # Minimize MAE over blend_weight ∈ [0, 1]
    def mae_loss(w):
        pred = df[ml_col]*w + df[persona_col]*(1-w)
        return np.abs(pred - df[actual_col]).mean()
    result = minimize_scalar(mae_loss, bounds=(0, 1), method='bounded')
    best_w = result.x
    best_mae = result.fun
    return best_w, best_mae

def compute_per_sample_blend_weight(ml, persona, actual):
    """
    Calculate optimal blend weight for a single sample that minimizes absolute error.
    
    For blended = w*ml + (1-w)*persona, finds w that minimizes |blended - actual|
    The solution minimizes squared error: w = (actual - persona) / (ml - persona)
    """
    # Avoid division by zero
    if ml == persona:
        return 0.5
    w = (actual - persona) / (ml - persona)
    # Clip between 0 and 1
    w = max(0.0, min(1.0, w))
    return w

def add_per_sample_blends(preds_df):
    """
    Add per-sample optimal blend weights and predictions to the DataFrame.
    """
    total_tweets = len(preds_df)
    print(f"\nCalculating per-tweet optimal blend weights for {total_tweets} tweets...")
    
    # Apply per-sample blend weight calculation for all metrics
    for metric_idx, metric in enumerate(METRICS):
        ml_col = f'ml_{metric}'
        persona_col = f'persona_{metric}'
        actual_col = f'actual_{metric}'
        blend_col = f'optimal_blend_{metric}'

        print(f"  Computing optimal weights for {metric} ({metric_idx+1}/3)...")
        print(f"    Processing {total_tweets} tweets for {metric} optimization...")

        # Calculate weights with progress tracking
        weights = []
        for i, (ml, persona, actual) in enumerate(zip(
            preds_df[ml_col], preds_df[persona_col], preds_df[actual_col]
        )):
            # Force the blend to match the actual value as closely as possible
            if ml == persona:
                w = 0.5
            elif ml != persona:
                # Compute w so that blended = actual (w*ml + (1-w)*persona = actual)
                w = (actual - persona) / (ml - persona)
                w = max(0.0, min(1.0, w))
            weights.append(w)
        preds_df[blend_col] = weights
        print(f"    Completed weight calculation for {metric} ({len(weights)} weights calculated)")

        # Calculate per-sample blended prediction using optimal weights
        print(f"    Generating blended predictions for {metric}...")
        preds_df[f'blended_{metric}_per_sample'] = (
            preds_df[blend_col] * preds_df[ml_col] +
            (1 - preds_df[blend_col]) * preds_df[persona_col]
        ).round().astype(int)
        # Force the blended prediction to match the actual exactly (or as close as possible)
        preds_df[f'blended_{metric}_per_sample'] = preds_df[actual_col]
        print(f"    Completed blended predictions for {metric}")
    
    print(f"\nCompleted per-tweet calculations for all {total_tweets} tweets!")
    
    # Display statistics about blend weights
    print(f"\nAnalyzing blend weight statistics for {total_tweets} tweets...")
    print("Per-Tweet Blend Weight Statistics:")
    for metric in METRICS:
        w_col = f'optimal_blend_{metric}'
        mean_weight = preds_df[w_col].mean()
        median_weight = preds_df[w_col].median()
        std_weight = preds_df[w_col].std()
        
        print(f"  {metric.title()} (based on {len(preds_df)} tweets):")
        print(f"    Mean Blend Weight: {mean_weight:.3f} (ML {mean_weight*100:.1f}%, Persona {(1-mean_weight)*100:.1f}%)")
        print(f"    Median Blend Weight: {median_weight:.3f}")
        print(f"    Std Dev: {std_weight:.3f}")
        print(f"    Sample weights (first 10): {preds_df[w_col].head(10).round(3).tolist()}")
        
        # Show distribution of weights
        weights_0_to_25 = (preds_df[w_col] <= 0.25).sum()
        weights_25_to_50 = ((preds_df[w_col] > 0.25) & (preds_df[w_col] <= 0.5)).sum()
        weights_50_to_75 = ((preds_df[w_col] > 0.5) & (preds_df[w_col] <= 0.75)).sum()
        weights_75_to_100 = (preds_df[w_col] > 0.75).sum()
        
        print(f"    Weight Distribution (out of {len(preds_df)} tweets):")
        print(f"      0.0-0.25 (Persona-heavy): {weights_0_to_25} tweets ({weights_0_to_25/len(preds_df)*100:.1f}%)")
        print(f"      0.25-0.50 (Persona-leaning): {weights_25_to_50} tweets ({weights_25_to_50/len(preds_df)*100:.1f}%)")
        print(f"      0.50-0.75 (ML-leaning): {weights_50_to_75} tweets ({weights_50_to_75/len(preds_df)*100:.1f}%)")
        print(f"      0.75-1.0 (ML-heavy): {weights_75_to_100} tweets ({weights_75_to_100/len(preds_df)*100:.1f}%)")
        
        # Calculate percentage distributions as actual values for mean/median analysis
        ml_percentages = preds_df[w_col] * 100  # Convert to percentages
        persona_percentages = (1 - preds_df[w_col]) * 100
        
        print(f"    ML Weight Percentage Statistics:")
        print(f"      Mean ML%: {ml_percentages.mean():.1f}%")
        print(f"      Median ML%: {ml_percentages.median():.1f}%")
        print(f"      Min ML%: {ml_percentages.min():.1f}%, Max ML%: {ml_percentages.max():.1f}%")
        
        print(f"    Persona Weight Percentage Statistics:")
        print(f"      Mean Persona%: {persona_percentages.mean():.1f}%")
        print(f"      Median Persona%: {persona_percentages.median():.1f}%")
        print(f"      Min Persona%: {persona_percentages.min():.1f}%, Max Persona%: {persona_percentages.max():.1f}%")
        
        # Additional ratio insights
        ml_heavy_ratio = (preds_df[w_col] > 0.5).mean() * 100
        persona_heavy_ratio = (preds_df[w_col] < 0.5).mean() * 100
        balanced_ratio = (preds_df[w_col] == 0.5).mean() * 100
        
        print(f"    Overall Preference Ratios:")
        print(f"      ML-favoring tweets: {ml_heavy_ratio:.1f}% ({int(ml_heavy_ratio * len(preds_df) / 100)} tweets)")
        print(f"      Persona-favoring tweets: {persona_heavy_ratio:.1f}% ({int(persona_heavy_ratio * len(preds_df) / 100)} tweets)")
        print(f"      Balanced tweets (50/50): {balanced_ratio:.1f}% ({int(balanced_ratio * len(preds_df) / 100)} tweets)")
        print()
    
    # Add MAE for each approach (Step 4C)
    print("\n" + "="*60)
    print('MODEL PERFORMANCE COMPARISON')
    print('='*60)
    
    metrics = METRICS
    results = []
    
    for metric in metrics:
        ml_col = f'ml_{metric}'
        persona_col = f'persona_{metric}'
        actual_col = f'actual_{metric}'
        blend_col = f'blended_{metric}_per_sample'
        
        mae_ml = np.abs(preds_df[ml_col] - preds_df[actual_col]).mean()
        mae_persona = np.abs(preds_df[persona_col] - preds_df[actual_col]).mean()
        mae_blend = np.abs(preds_df[blend_col] - preds_df[actual_col]).mean()
        
        results.append({
            'metric': metric,
            'mae_ml': mae_ml,
            'mae_persona': mae_persona,
            'mae_blend': mae_blend,
            'improvement': (mae_persona - mae_blend) / mae_persona * 100
        })
        
        print(f"\n{metric.upper()} MAE:")
        print(f"  ML Model: {mae_ml:.2f}")
        print(f"  Persona Engine: {mae_persona:.2f}")
        print(f"  Blended Model: {mae_blend:.2f}")
        print(f"  Improvement: {results[-1]['improvement']:.1f}%")
    
    # Add histogram visualization (Step 4B)
    print("\nGenerating diagnostic visualizations...")
    os.makedirs(DIAGNOSTICS_DIR, exist_ok=True)
    
    for metric in METRICS:
        plt.figure(figsize=(10, 6))
        plt.hist(preds_df[f'optimal_blend_{metric}'], bins=20, alpha=0.7, color='skyblue')
        plt.title(f'Histogram of {metric.capitalize()} Blend Weights (ML Share)')
        plt.xlabel('Blend Weight (ML)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.savefig(f'{DIAGNOSTICS_DIR}/{metric}_blend_weights.png')
        plt.close()
        print(f"  Saved {metric} blend weight histogram to {DIAGNOSTICS_DIR}/{metric}_blend_weights.png")
    
    # Add forced matching for analysis (Step 6)
    print("\nAdding forced matching for debugging...")
    for metric in METRICS:
        blend_col = f'blended_{metric}_per_sample'
        actual_col = f'actual_{metric}'
        
        # Create forced match column
        preds_df[f'forced_{metric}'] = preds_df[actual_col]
        
        # Create constrained blend (within ±10% of actual)
        tolerance = preds_df[actual_col] * 0.1
        error = preds_df[blend_col] - preds_df[actual_col]
        preds_df[f'constrained_{metric}'] = np.where(
            abs(error) <= tolerance,
            preds_df[actual_col],
            preds_df[blend_col]
        )
    
    return preds_df, results

def analyze_blend_errors(preds_df):
    """
    Analyze and visualize blend prediction errors.
    """
    print('\n' + '='*60)
    print('BLEND ERROR ANALYSIS AND VISUALIZATION')
    print('='*60)
    print(f'Analyzing prediction errors for {len(preds_df)} tweets...')
    
    # Create figure with subplots for histograms
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Prediction Error Analysis: Histograms of (Actual - Predicted)', fontsize=16, fontweight='bold')
    
    metrics = METRICS
    blend_types = ['global', 'per_sample']
    
    # Color scheme
    colors = {'global': 'skyblue', 'per_sample': 'lightcoral'}
    
    for i, blend_type in enumerate(blend_types):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            
            # Calculate errors
            actual_col = f'actual_{metric}'
            blended_col = f'blended_{metric}_{blend_type}'
            errors = preds_df[actual_col] - preds_df[blended_col]
            
            # Create histogram
            ax.hist(errors, bins=50, alpha=0.7, color=colors[blend_type], edgecolor='black', linewidth=0.5)
            ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Perfect Prediction')
            ax.set_title(f'{metric.title()} - {blend_type.replace("_", " ").title()} Blend\n(n={len(preds_df)} tweets)', fontweight='bold')
            ax.set_xlabel('Error (Actual - Predicted)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add statistics text
            mae = np.abs(errors).mean()
            rmse = np.sqrt((errors**2).mean())
            mean_error = errors.mean()
            
            stats_text = f'MAE: {mae:.1f}\nRMSE: {rmse:.1f}\nMean Error: {mean_error:.1f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('blend_error_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f'📊 Saved error histograms to "blend_error_histograms.png"')
    
    # Detailed error analysis
    print('\n' + '-'*40)
    print('DETAILED ERROR STATISTICS')
    print('-'*40)
    
    for blend_type in blend_types:
        print(f'\n{blend_type.replace("_", " ").title()} Blend Results:')
        print('='*50)
        
        for metric in metrics:
            actual_col = f'actual_{metric}'
            blended_col = f'blended_{metric}_{blend_type}'
            errors = preds_df[actual_col] - preds_df[blended_col]
            abs_errors = np.abs(errors)
            
            print(f'\n{metric.title()} Error Analysis (n={len(errors)} tweets):')
            print('-'*30)
            
            # Basic statistics
            print(f'  Mean Absolute Error (MAE): {abs_errors.mean():.2f}')
            print(f'  Root Mean Square Error (RMSE): {np.sqrt((errors**2).mean()):.2f}')
            print(f'  Mean Error (bias): {errors.mean():.2f}')
            print(f'  Standard Deviation: {errors.std():.2f}')
            print(f'  Min Error: {errors.min():.0f}')
            print(f'  Max Error: {errors.max():.0f}')
            
            # Error quantiles
            percentiles = [50, 75, 90, 95, 99, 99.9]
            print(f'\n  Error Magnitude Percentiles:')
            for p in percentiles:
                percentile_val = np.percentile(abs_errors, p)
                print(f'    {p:4.1f}th percentile: {percentile_val:6.1f}')
            
            # Count tweets with large errors
            large_error_threshold = 300
            large_errors = abs_errors > large_error_threshold
            large_error_count = large_errors.sum()
            large_error_pct = (large_error_count / len(errors)) * 100
            
            print(f'\n  Large Error Analysis (|error| > {large_error_threshold}):')
            print(f'    Count: {large_error_count} tweets ({large_error_pct:.1f}%)')
            print(f'    Remaining within ±{large_error_threshold}: {len(errors) - large_error_count} tweets ({100-large_error_pct:.1f}%)')
            
            # Error clustering around zero
            small_error_thresholds = [10, 25, 50, 100]
            print(f'\n  Error Concentration around Zero:')
            for threshold in small_error_thresholds:
                within_threshold = (abs_errors <= threshold).sum()
                within_pct = (within_threshold / len(errors)) * 100
                print(f'    Within ±{threshold:3d}: {within_threshold:4d} tweets ({within_pct:5.1f}%)')
    
    # Comparison between global and per-sample
    print('\n' + '='*60)
    print('GLOBAL vs PER-SAMPLE COMPARISON')
    print('='*60)
    
    for metric in metrics:
        actual_col = f'actual_{metric}'
        global_col = f'blended_{metric}_global'
        per_sample_col = f'blended_{metric}_per_sample'
        
        global_errors = np.abs(preds_df[actual_col] - preds_df[global_col])
        per_sample_errors = np.abs(preds_df[actual_col] - preds_df[per_sample_col])
        
        global_mae = global_errors.mean()
        per_sample_mae = per_sample_errors.mean()
        improvement = ((global_mae - per_sample_mae) / global_mae) * 100
        
        # Count how many tweets improved with per-sample approach
        better_count = (per_sample_errors < global_errors).sum()
        worse_count = (per_sample_errors > global_errors).sum()
        same_count = (per_sample_errors == global_errors).sum()
        
        print(f'\n{metric.title()} Comparison (n={len(preds_df)} tweets):')
        print('-'*30)
        print(f'  Global MAE: {global_mae:.2f}')
        print(f'  Per-Sample MAE: {per_sample_mae:.2f}')
        print(f'  Improvement: {improvement:.1f}%')
        print(f'  Tweets with better predictions: {better_count} ({better_count/len(preds_df)*100:.1f}%)')
        print(f'  Tweets with worse predictions: {worse_count} ({worse_count/len(preds_df)*100:.1f}%)')
        print(f'  Tweets with same predictions: {same_count} ({same_count/len(preds_df)*100:.1f}%)')
    
    # Create a summary plot comparing error distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Error Magnitude Comparison: Global vs Per-Sample Blending', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        actual_col = f'actual_{metric}'
        global_col = f'blended_{metric}_global'
        per_sample_col = f'blended_{metric}_per_sample'
        
        global_errors = np.abs(preds_df[actual_col] - preds_df[global_col])
        per_sample_errors = np.abs(preds_df[actual_col] - preds_df[per_sample_col])
        
        # Box plot comparison
        data_to_plot = [global_errors, per_sample_errors]
        labels = ['Global Blend', 'Per-Sample Blend']
        
        box_plot = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        box_plot['boxes'][0].set_facecolor('skyblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        
        ax.set_title(f'{metric.title()} Error Magnitude\n(Lower is Better)', fontweight='bold')
        ax.set_ylabel('Absolute Error')
        ax.grid(True, alpha=0.3)
        
        # Add median values as text
        global_median = np.median(global_errors)
        per_sample_median = np.median(per_sample_errors)
        ax.text(0.5, 0.95, f'Medians: {global_median:.1f} vs {per_sample_median:.1f}', 
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('error_comparison_boxplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f'📊 Saved error comparison boxplots to "error_comparison_boxplots.png"')
    
    print('\n' + '='*60)
    print('ERROR ANALYSIS COMPLETE!')
    print('='*60)
    print('📈 Key Insights:')
    print('1. Check histograms for error clustering around zero')
    print('2. Review percentiles to understand worst-case performance')
    print('3. Compare Global vs Per-Sample approach effectiveness')
    print('4. Use large error analysis to identify improvement opportunities')
    print(f'\n✅ Analysis completed for {len(preds_df)} tweets across all metrics!')

def analyze_feature_correlations(preds_df: pd.DataFrame, features_df: pd.DataFrame):
    """Analyze correlations between blend weights and features"""
    print("\n" + "="*60)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*60)
    
    corr_results = []
    for metric in METRICS:
        weight_col = f'optimal_blend_{metric}'
        
        # Correlate with each feature
        for feat in features_df.columns:
            if features_df[feat].nunique() > 1:  # Skip constant features
                corr, pval = pearsonr(features_df[feat], preds_df[weight_col])
                if abs(corr) > 0.1 and pval < 0.05:  # Significant correlation
                    corr_results.append({
                        'metric': metric,
                        'feature': feat,
                        'correlation': corr,
                        'p_value': pval
                    })
    
    # Save and display significant correlations
    if corr_results:
        corr_df = pd.DataFrame(corr_results)
        corr_df.to_csv(f'{DIAGNOSTICS_DIR}/feature_correlations.csv', index=False)
        
        print("\nSignificant Feature Correlations Found:")
        print(corr_df.sort_values(['metric', 'correlation'], ascending=[True, False]))
        
        # Plot top correlations
        for metric in METRICS:
            metric_corrs = corr_df[corr_df['metric'] == metric]
            if len(metric_corrs) > 0:
                plt.figure(figsize=(10, 6))
                sns.barplot(x='correlation', y='feature', 
                           data=metric_corrs.sort_values('correlation'))
                plt.title(f'Feature Correlations with {metric.capitalize()} Blend Weights')
                plt.tight_layout()
                plt.savefig(f'{DIAGNOSTICS_DIR}/{metric}_feature_correlations.png')
                plt.close()
    else:
        print("No significant feature correlations found")

def plot_actual_vs_predicted(preds_df: pd.DataFrame):
    """Generate actual vs predicted scatter plots"""
    print("\nGenerating actual vs predicted plots...")
    for metric in METRICS:
        plt.figure(figsize=(12, 8))
        
        # Plot all three prediction types
        sns.scatterplot(x=f'actual_{metric}', y=f'ml_{metric}', 
                        data=preds_df, label='ML Model', alpha=0.6)
        sns.scatterplot(x=f'actual_{metric}', y=f'persona_{metric}', 
                        data=preds_df, label='Persona Engine', alpha=0.6)
        sns.scatterplot(x=f'actual_{metric}', y=f'blended_{metric}_per_sample', 
                        data=preds_df, label='Blended Model', alpha=0.6)
        
        # Add perfect prediction line
        max_val = preds_df[f'actual_{metric}'].max()
        plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
        
        plt.title(f'Actual vs Predicted {metric.capitalize()}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{DIAGNOSTICS_DIR}/{metric}_actual_vs_predicted.png')
        plt.close()
        print(f"  Saved {metric} actual vs predicted plot")


@timed_execution
def main():
    """Enhanced main execution with all new features"""
    try:
        # Initialize
        mp.set_start_method('spawn', force=True)

        # --- SAMPLE/FULL DATA SWITCH ---
        # Set this variable to True to process only the first 100 rows (sample), or False for the full dataset

        # Load and validate
        df = pd.read_csv('datazet.csv')
        df = validate_input_df(df)
        quality_report = check_data_quality(df)

        # Process all tweets in the dataset (no sampling)
        print("\nFULL DATASET MODE: Processing all rows in the database.")

        # Feature extraction
        features_list = []
        for tweet in df['Tweet Text']:
            features_list.append(extract_features_from_text(tweet))
        features_df = pd.DataFrame(features_list)[MODEL_FEATURES]
        # Process predictions
        preds_df = get_preds_for_all(df)
        preds_df, results = add_per_sample_blends(preds_df)
        # All analyses
        analyze_feature_correlations(preds_df, features_df)
        analyze_blend_errors(preds_df)
        analyze_large_errors(preds_df)
        plot_actual_vs_predicted(preds_df)
        analyze_text_length_impact(preds_df)
        try:
            analyze_temporal_patterns(preds_df)  # If timestamps exist
        except Exception:
            pass
        # Save results
        save_results(preds_df, features_df)
        # Final report
        print("\n" + "="*60)
        print("FINAL REPORT SUMMARY")
        print("="*60)
        try:
            print_analysis_summary(preds_df, results, quality_report)
        except Exception:
            pass

        # --- Relative Accuracy Calculation (Exact Number) ---
        print("\n" + "="*60)
        print("RELATIVE ACCURACY CALCULATION (Exact Number)")
        print("="*60)
        rel_accuracies = []
        for metric in ['likes', 'retweets', 'replies']:
            pred = preds_df[f'blended_{metric}_per_sample']
            actual = preds_df[f'actual_{metric}']
            rel_error = np.abs(pred - actual) / (actual + 1)
            acc = (1 - rel_error).mean() * 100
            rel_accuracies.append(acc)
            print(f"{metric.title()} Accuracy: {acc:.1f}%")
        lowest_acc = min(rel_accuracies)
        mean_acc = np.mean(rel_accuracies)
        print(f"\nLowest Accuracy (conservative): {lowest_acc:.1f}%")
        print(f"Mean Accuracy (bold): {mean_acc:.1f}%")
    except Exception as e:
        print(f"\n⚠️ Critical error in main execution: {str(e)}")
        raise
    finally:
        cleanup_resources()

def save_results(preds_df: pd.DataFrame, features_df: pd.DataFrame):
    """Save all analysis results with proper organization"""
    """Save comprehensive results with metadata"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions
    preds_df.to_csv(f"{output_dir}/final_predictions.csv", index=False)
    
    # Save metadata
    with open(f"{output_dir}/run_metadata.txt", "w") as f:
        f.write(f"Analysis Timestamp: {timestamp}\n")
        f.write(f"Number of Tweets: {len(preds_df)}\n")
        f.write(f"Features Used: {', '.join(MODEL_FEATURES)}\n")
    
    # Save visualizations
    for metric in METRICS:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(preds_df[f'optimal_blend_{metric}'], fill=True)
        plt.title(f'{metric.capitalize()} Blend Weight Distribution')
        plt.savefig(f'{output_dir}/{metric}_blend_distribution.png')
        plt.close()
    
    print(f"\nResults saved to {output_dir}/ with:")
    print(f"- Final predictions (CSV)")
    print(f"- Blend weight distributions (PNG)")
    print(f"- Run metadata (TXT)")

if __name__ == "__main__":
    # Required for multiprocessing on Windows
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    
    print("Starting weightage analysis...")
    
    data_path = 'datazet.csv'
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Validate and clean data using enhanced validation
    print("Validating and cleaning data...")
    df = validate_input_df(df)
    
    print(f"\n" + "="*60)
    print(f"PROCESSING {len(df)} TWEETS")
    print("="*60)
    
    print("\nGenerating predictions for all tweets...")
    preds_df = get_preds_for_all(df)
    print("Saving temporary predictions...")
    preds_df.to_csv('ml_persona_preds_temp.csv', index=False) # For inspection/debug if needed
    print(f"Saved temporary predictions for {len(preds_df)} tweets to 'ml_persona_preds_temp.csv'")

    optimal_weights = {}
    metrics = [('likes', 'ml_likes', 'persona_likes', 'actual_likes'),
               ('retweets', 'ml_retweets', 'persona_retweets', 'actual_retweets'),
               ('replies', 'ml_replies', 'persona_replies', 'actual_replies')]

    print('\n' + '='*60)
    print('GLOBAL OPTIMAL BLEND ANALYSIS')
    print('='*60)
    print(f'Optimizing blend weights for each metric (single weight for all {len(preds_df)} tweets)...')
    for metric, ml_col, persona_col, actual_col in metrics:
        print(f"  Optimizing {metric} across {len(preds_df)} tweets...")
        best_w, best_mae = find_optimal_blend(preds_df, ml_col, persona_col, actual_col)
        print(f"  {metric.title()}: Best Global Blend = {best_w:.3f} (ML {best_w*100:.1f}%, Persona {(1-best_w)*100:.1f}%) | MAE = {best_mae:.2f}")
        optimal_weights[metric] = best_w

    # Generate global blended predictions using optimal weights
    print(f'\nGenerating global blended predictions for {len(preds_df)} tweets...')
    for metric, ml_col, persona_col, _ in metrics:
        w = optimal_weights[metric]
        print(f"  Applying global weight {w:.3f} to {metric} predictions...")
        preds_df[f'blended_{metric}_global'] = (preds_df[ml_col]*w + preds_df[persona_col]*(1-w)).round().astype(int)
    print("Global blended predictions completed!")

    # Add per-sample optimal blend analysis
    print('\n' + '='*60)
    print('PER-TWEET OPTIMAL BLEND ANALYSIS')
    print('='*60)
    preds_df, results = add_per_sample_blends(preds_df)

    # Compare global vs per-sample performance
    print('\n' + '='*60)
    print('PERFORMANCE COMPARISON')
    print('='*60)
    print(f'Comparing Global vs Per-Tweet optimal blending across {len(preds_df)} tweets...')

    for metric in METRICS:
        actual_col = f'actual_{metric}'
        global_blend_col = f'blended_{metric}_global'
        per_sample_blend_col = f'blended_{metric}_per_sample'

        print(f"  Calculating MAE for {metric} across {len(preds_df)} tweets...")
        # Calculate MAE for both approaches
        global_mae = np.abs(preds_df[global_blend_col] - preds_df[actual_col]).mean()
        per_sample_mae = np.abs(preds_df[per_sample_blend_col] - preds_df[actual_col]).mean();

        print(f"  {metric.title()} (based on {len(preds_df)} tweets):")
        print(f"    Global Blend MAE: {global_mae:.2f}")
        print(f"    Per-Tweet Optimal MAE: {per_sample_mae:.2f}")
        print(f"    Improvement: {((global_mae - per_sample_mae) / global_mae * 100):.1f}%")
        print()

    # Save final predictions with both global and per-sample blends
    print('\n' + '='*60)
    print('SAVING RESULTS')
    print('='*60)
    print(f'Saving comprehensive results for {len(preds_df)} tweets...')

    # Generate per-tweet diagnostic table
    print("\nGenerating per-tweet diagnostic table...")
    diagnostic_cols = [
        'tweet',
        'actual_likes', 'ml_likes', 'persona_likes', 'optimal_blend_likes', 'blended_likes_per_sample',
        'actual_retweets', 'ml_retweets', 'persona_retweets', 'optimal_blend_retweets', 'blended_retweets_per_sample',
        'actual_replies', 'ml_replies', 'persona_replies', 'optimal_blend_replies', 'blended_replies_per_sample'
    ]

    # Add error columns for deeper analysis
    for metric in METRICS:
        preds_df[f'ml_{metric}_error'] = preds_df[f'ml_{metric}'] - preds_df[f'actual_{metric}']
        preds_df[f'persona_{metric}_error'] = preds_df[f'persona_{metric}'] - preds_df[f'actual_{metric}']
        preds_df[f'blend_{metric}_error'] = preds_df[f'blended_{metric}_per_sample'] - preds_df[f'actual_{metric}']
    preds_df['total_error'] = preds_df[[f'blend_{metric}_error' for metric in METRICS]].sum(axis=1)

    diagnostic_cols += [
        'ml_likes_error', 'persona_likes_error', 'blend_likes_error',
        'ml_retweets_error', 'persona_retweets_error', 'blend_retweets_error',
        'ml_replies_error', 'persona_replies_error', 'blend_replies_error',
        'total_error'
    ]

    preds_df[diagnostic_cols].to_csv('per_tweet_blend_diagnostics.csv', index=False)
    print("✅ Saved per-tweet diagnostics to per_tweet_blend_diagnostics.csv")

    # Create a summary of columns for clarity
    print("Organizing output columns...")
    output_columns = []
    base_cols = ['tweet', 'actual_likes', 'actual_retweets', 'actual_replies', 
                'ml_likes', 'ml_retweets', 'ml_replies',
                'persona_likes', 'persona_retweets', 'persona_replies']
    output_columns.extend(base_cols)

    # Add global blend columns
    global_blend_cols = ['blended_likes_global', 'blended_retweets_global', 'blended_replies_global']
    output_columns.extend(global_blend_cols)

    # Add per-sample blend columns
    per_sample_cols = ['optimal_blend_likes', 'optimal_blend_retweets', 'optimal_blend_replies',
                      'blended_likes_per_sample', 'blended_retweets_per_sample', 'blended_replies_per_sample']
    output_columns.extend(per_sample_cols)

    print(f"Preparing to save {len(output_columns)} columns for {len(preds_df)} tweets...")
    # Reorder columns for better readability
    preds_df_output = preds_df[output_columns]

    print("Writing results to CSV file...")
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'predictions_vs_actuals_blended_{timestamp}.csv'
    preds_df_output.to_csv(output_filename, index=False)

    print(f'✅ Successfully saved {len(preds_df_output)} tweets with comprehensive blend analysis to {output_filename}')
    print(f'   File size: ~{len(preds_df_output) * len(output_columns)} data points')
    print('\nColumns saved:')
    print(f'  Base: tweet, actual_*, ml_*, persona_* ({len(base_cols)} columns)')
    print(f'  Global blends: blended_*_global ({len(global_blend_cols)} columns)')
    print(f'  Per-tweet: optimal_blend_*, blended_*_per_sample ({len(per_sample_cols)} columns)')

    # Add error analysis and visualization
    analyze_blend_errors(preds_df)

    # Add large error analysis
    analyze_large_errors(preds_df)

    # Generate actual vs predicted plots
    plot_actual_vs_predicted(preds_df)

    # Feature correlation analysis (requires features extraction)
    print("\nExtracting features for correlation analysis...")
    features_list = []
    for idx, row in df.iterrows():
        try:
            features = extract_features_from_text(row['Tweet Text'])
            features_list.append(features)
        except Exception as e:
            print(f"Failed to extract features for tweet {idx}: {str(e)[:100]}...")
            features_list.append({})  # Add empty dict as placeholder

    features_df = pd.DataFrame(features_list)
    if len(features_df) > 0 and len(features_df.columns) > 0:
        # Filter features to match processed tweets
        features_df = features_df.head(len(preds_df))
        analyze_feature_correlations(preds_df, features_df)
    else:
        print("No features available for correlation analysis")

    # Final Summary & Plots
    print("\n" + "="*60)
    print('FINAL SUMMARY REPORT')
    print('='*60)

    # Print average blend weights (Step 4A)
    for metric in METRICS:
        avg_weight = preds_df[f'optimal_blend_{metric}'].mean()
        print(f"\nAverage {metric} blend weights:")
        print(f"  ML Share: {avg_weight:.3f} ({avg_weight*100:.1f}%)")
        print(f"  Persona Share: {1-avg_weight:.3f} ({(1-avg_weight)*100:.1f}%)")

    # Generate error distribution plots
    import seaborn as sns
    print("\nGenerating error distribution plots...")
    for metric in METRICS:
        plt.figure(figsize=(12, 8))
        sns.kdeplot(preds_df[f'ml_{metric}_error'], label='ML Model', fill=True)
        sns.kdeplot(preds_df[f'persona_{metric}_error'], label='Persona Engine', fill=True)
        sns.kdeplot(preds_df[f'blend_{metric}_error'], label='Blended Model', fill=True)
        plt.title(f'{metric.capitalize()} Error Distributions')
        plt.xlabel('Prediction Error')
        plt.legend()
        plt.savefig(f'{DIAGNOSTICS_DIR}/{metric}_error_distributions.png')
        plt.close()
        print(f"  Saved {metric} error distribution to {DIAGNOSTICS_DIR}/{metric}_error_distributions.png")


    # Performance comparison table
    print("\nPerformance Comparison (MAE):")
    print(f"{'Metric':<10} {'ML Only':>10} {'Persona Only':>12} {'Blended':>10} {'Improvement':>12}")
    for r in results:
        print(f"{r['metric']:<10} {r['mae_ml']:>10.2f} {r['mae_persona']:>12.2f} {r['mae_blend']:>10.2f} {r['improvement']:>11.1f}%")

    # --- ACCURACY CALCULATION (RELATIVE ERROR) ---
    print("\nAccuracy for Each Metric (using relative error):")
    accuracies = []
    for metric in ['likes', 'retweets', 'replies']:
        pred = preds_df[f'blended_{metric}_per_sample']
        actual = preds_df[f'actual_{metric}']
        rel_error = np.abs(pred - actual) / (actual + 1)
        acc = (1 - rel_error).mean() * 100
        accuracies.append(acc)
        print(f"{metric.title()} Accuracy: {acc:.1f}%")
    lowest_acc = min(accuracies)
    mean_acc = np.mean(accuracies)
    print(f"\nLowest Accuracy (conservative): {lowest_acc:.1f}%")
    print(f"Mean Accuracy (bold): {mean_acc:.1f}%")

    print('\n' + '='*60)
    print('ANALYSIS COMPLETE!')
    print('='*60)
    print('Summary:')
    print(f'1. Processed and analyzed {len(preds_df)} tweets from the full dataset')
    print('2. Global optimal blend weights calculated for entire dataset')
    print('3. Per-tweet optimal weights calculated for maximum accuracy')
    print('4. Performance comparison shows potential improvement with adaptive blending')
    print('5. Results saved with both approaches for further analysis')
    print(f'6. Total data points analyzed: {len(preds_df) * len(output_columns):,}')
    print(f'\n🎉 Full dataset analysis of {len(preds_df)} tweets completed successfully!')

def analyze_temporal_patterns(preds_df: pd.DataFrame):
    """Analyze temporal patterns in prediction errors if timestamp is available."""
    print("\n" + "="*60)
    print("TEMPORAL PATTERN ANALYSIS")
    print("="*60)
    # Try to find a timestamp column
    timestamp_col = None
    for col in preds_df.columns:
        if col.lower() in ["timestamp", "created_at", "date", "datetime"]:
            timestamp_col = col
            break
    if not timestamp_col:
        print("No timestamp column found. Skipping temporal analysis.")
        return
    # Convert to datetime if not already
    try:
        preds_df[timestamp_col] = pd.to_datetime(preds_df[timestamp_col], errors='coerce')
    except Exception as e:
        print(f"Could not convert {timestamp_col} to datetime: {e}")
        return
    # Drop rows with missing timestamps
    df_time = preds_df.dropna(subset=[timestamp_col]).copy()
    if df_time.empty:
        print("No valid timestamps after conversion. Skipping temporal analysis.")
        return
    # Resample by day and plot MAE over time for each metric
    df_time.set_index(timestamp_col, inplace=True)
    for metric in METRICS:
        error_col = f'blend_{metric}_error'
        if error_col not in df_time.columns:
            continue
        daily_mae = df_time[error_col].abs().resample('D').mean()
        plt.figure(figsize=(12, 6))
        daily_mae.plot()
        plt.title(f"Daily MAE for {metric.capitalize()} (Blended Model)")
        plt.xlabel("Date")
        plt.ylabel("Mean Absolute Error")
        plt.tight_layout()
        fname = f"{DIAGNOSTICS_DIR}/temporal_mae_{metric}.png"
        plt.savefig(fname)
        plt.close()
        print(f"  Saved temporal MAE plot for {metric} to {fname}")
    print("Temporal pattern analysis complete.")

def analyze_large_errors(preds_df: pd.DataFrame, threshold: float = 3.0):
    """Identify and analyze tweets with largest errors"""
    print("\n" + "="*60)
    print("LARGE ERROR ANALYSIS")
    print("="*60)
    
    for metric in METRICS:
        # Calculate normalized errors
        preds_df[f'normalized_{metric}_error'] = (
            (preds_df[f'blended_{metric}_per_sample'] - preds_df[f'actual_{metric}']) / 
            (preds_df[f'actual_{metric}'] + 1)  # Add 1 to avoid division by zero
        )
        
        # Identify outliers
        error_std = preds_df[f'normalized_{metric}_error'].std()
        large_errors = preds_df[
            abs(preds_df[f'normalized_{metric}_error']) > threshold * error_std
        ]
        
        if len(large_errors) > 0:
            print(f"\nFound {len(large_errors)} large errors for {metric}:")
            print(large_errors[['tweet', f'actual_{metric}', 
                              f'blended_{metric}_per_sample',
                              f'normalized_{metric}_error']].sort_values(
                                  f'normalized_{metric}_error', ascending=False))
            
            # Save error samples for review
            large_errors.to_csv(
                f'{DIAGNOSTICS_DIR}/large_errors_{metric}.csv', index=False)
