# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# Load feature-engineered data from Phase 1
input_file = 'features_datazet.csv'  # Output from Phase 1
df = pd.read_csv(input_file)

# Check the data types and preview
print("Data types:")
print(df[['likes', 'retweets', 'replies']].dtypes)
print("\nFirst few values:")
print(df[['likes', 'retweets', 'replies']].head())

# --- 1. Prepare Regression Targets ---
# Function to convert text formats like "1K", "2.5M" to numbers
def convert_engagement_text(value):
    if pd.isna(value) or value == '':
        return 0
    
    # Convert to string and clean
    value = str(value).strip().upper()
    
    # If it's already a number, return it
    try:
        return float(value)
    except ValueError:
        pass
    
    # Handle K, M, B suffixes
    multipliers = {'K': 1000, 'M': 1000000, 'B': 1000000000}
    
    for suffix, multiplier in multipliers.items():
        if value.endswith(suffix):
            try:
                number = float(value[:-1])
                return number * multiplier
            except ValueError:
                return 0
    
    # If no suffix, try to extract numbers
    import re
    numbers = re.findall(r'\d+\.?\d*', value)
    if numbers:
        return float(numbers[0])
    
    return 0

# Convert engagement metrics to numeric, handling text formats
for metric in ['likes', 'retweets', 'replies']:
    print(f"\nProcessing {metric}...")
    print(f"Sample values before conversion: {df[metric].head().tolist()}")
    
    # Apply conversion function
    df[metric] = df[metric].apply(convert_engagement_text)
    
    # Ensure no negative values
    df[metric] = df[metric].abs()
    
    print(f"Sample values after conversion: {df[metric].head().tolist()}")
    
    # Log transformation for regression targets
    df[f'log_{metric}'] = np.log1p(df[metric])

print(f"\nAfter conversion - Data types:")
print(df[['likes', 'retweets', 'replies']].dtypes)

# --- 2. Prepare Virality Classification Target ---
# Create combined engagement score (adjust weights as needed)
weights = {
    'likes': 0.6,
    'retweets': 0.3,
    'replies': 0.1
}
df['engagement_score'] = (
    weights['likes'] * df['likes'] + 
    weights['retweets'] * df['retweets'] + 
    weights['replies'] * df['replies']
)

# Set viral threshold (top 5%)
VIRAL_QUANTILE = 0.95
viral_threshold = df['engagement_score'].quantile(VIRAL_QUANTILE)
df['is_viral'] = (df['engagement_score'] >= viral_threshold).astype(int)

# --- Enhanced Virality Tiers ---
# Define virality tiers with multiple levels
tier_thresholds = [0.70, 0.85, 0.95, 0.99]  # Percentile cutoffs
tier_labels = ['Low', 'Medium', 'High', 'Viral', 'Super Viral']

# Calculate percentile values
threshold_values = df['engagement_score'].quantile(tier_thresholds).tolist()

# Create tiers
conditions = [
    df['engagement_score'] < threshold_values[0],
    (df['engagement_score'] >= threshold_values[0]) & (df['engagement_score'] < threshold_values[1]),
    (df['engagement_score'] >= threshold_values[1]) & (df['engagement_score'] < threshold_values[2]),
    (df['engagement_score'] >= threshold_values[2]) & (df['engagement_score'] < threshold_values[3]),
    df['engagement_score'] >= threshold_values[3]
]

df['virality_tier'] = np.select(conditions, tier_labels, default='Low')

# --- 3. Save Final Dataset ---
# Keep all columns including the converted likes, retweets, replies
output_file = 'tweet_data_with_tiers.csv'
df.to_csv(output_file, index=False)

# --- Print Summary Report ---
viral_percentage = df['is_viral'].mean() * 100
print("\n" + "="*50)
print("PHASE 2 COMPLETE: TARGET ENGINEERING REPORT")
print("="*50)
print(f"Input file: {input_file}")
print(f"Output file: {output_file}")
print(f"\nTarget Variables Created:")
print(f"- Regression targets: log_likes, log_retweets, log_replies")
print(f"- Classification target: is_viral")
print(f"- Enhanced tiers: virality_tier")
print(f"\nVirality Threshold Analysis:")
print(f"- Engagement score threshold: {viral_threshold:.2f}")
print(f"- Viral tweets: {df['is_viral'].sum()} tweets ({viral_percentage:.1f}%)")

# Print tier distribution
tier_distribution = df['virality_tier'].value_counts(normalize=True).sort_index()
print("\nVirality Tier Distribution:")
for tier, percentage in tier_distribution.items():
    print(f"- {tier}: {percentage:.1%}")

print("\nFirst 3 rows preview:")
print(df.iloc[:3, -7:])  # Show last 7 columns (targets + new features)