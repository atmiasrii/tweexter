import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from features import extract_features_from_text, MODEL_FEATURES, select_features
from persona_engine import load_personas, optimize_and_predict, aggregate_engagement
from calibrator import ViralCalibrator

# --- Specialized Engagement Models ---
from retweet_model import RetweetPredictor
from reply_model import ReplyPredictor
# --- Initialize specialized predictors (do once at startup) ---
retweet_predictor = RetweetPredictor()
reply_predictor = ReplyPredictor()
# Use MODEL_FEATURES from features.py (27 features in correct order)
reg_seeds = [42, 99, 123, 456, 789]

# Load likes ensemble models
reg_models_likes = [joblib.load(f'regression_xgb_seed_{seed}.pkl') for seed in reg_seeds]
likes_scaler = joblib.load('likes_log_scaler.pkl')

# Load retweets ensemble models
reg_models_rt = [joblib.load(f'regression_rt_xgb_seed_{seed}.pkl') for seed in reg_seeds]
rt_scaler = joblib.load('retweets_log_scaler.pkl')

# Load replies ensemble models
reg_models_reply = [joblib.load(f'regression_reply_xgb_seed_{seed}.pkl') for seed in reg_seeds]
reply_scaler = joblib.load('replies_log_scaler.pkl')

def safe_expm1(x):
    x = np.clip(x, None, 20)
    return np.expm1(x)

PERSONAS = load_personas("personas.json")
DEMO_WEIGHTS = {}
CALIBRATOR = ViralCalibrator("ready_datazet.csv")

def calculate_confidence(ml_prediction, optimization_results):
    # Example: combine ML and persona confidence factors
    ml_conf = float(ml_prediction.get('confidence', '0').replace('%','')) / 100 if 'confidence' in ml_prediction else 0.7
    persona_conf = 0.7 + 0.3 * (optimization_results['best_variant']['elo'] / 1600)
    return f"{min(0.95, ml_conf * persona_conf) * 100:.0f}%"

def predict_virality(tweet):
    # 1. Extract features for ML models (27 features)
    features = extract_features_from_text(tweet)
    
    # 2. ML prediction - Filter to only MODEL_FEATURES in correct order
    features_df = pd.DataFrame([features])
    features_df = features_df[MODEL_FEATURES]  # Select only 27 model features, in order
    
    # Likes prediction
    likes_preds = [m.predict(features_df)[0] for m in reg_models_likes]
    likes_base_pred = np.mean(likes_preds)
    likes_pred_log = likes_base_pred * likes_scaler['std'] + likes_scaler['mean']
    likes = int(safe_expm1(likes_pred_log))


    # --- Use specialized models for retweets and replies ---
    retweets = int(retweet_predictor.predict(features))
    replies = int(reply_predictor.predict(features))

    # --- Prediction validation ---
    if retweets > 10 * likes:
        retweets = int(likes * 0.1)  # Sanity check


    # Replies should not exceed likes * 1.5
    replies = min(replies, int(likes * 1.5))

    ml_prediction = {
        'likes': likes,
        'retweets': retweets,
        'replies': replies,
        'confidence': '80%',
    }
    
    # 3. Persona simulation (independent - no ML features needed)
    persona_result = aggregate_engagement(tweet, PERSONAS)  # Just passes tweet text
    persona_likes = persona_result['persona_likes']
    
    # 4. Calibrate using viral feature subset (legacy compatibility)
    viral_features = select_features({**features,
                                      'ml_prediction': ml_prediction['likes'],
                                      'persona_prediction': persona_likes})
    result = CALIBRATOR.predict(viral_features)

    # Three-tier prediction
    if result['virality_tier'] == 'High':
        final_pred = result['predicted_likes'] * 1.5
    else:
        final_pred = result['predicted_likes']

    confidence = f"{int(result['virality_prob'] * 100)}%"

    return {
        'likes': int(final_pred),
        'retweets': retweets,
        'replies': replies,
        'confidence': confidence,
        'ml_prediction': ml_prediction['likes'],
        'persona_prediction': persona_likes,
        'tier': result['virality_tier'],
        'top_demographics': persona_result.get('top_segments', [])[:3],
        'improvement_analysis': None
    }

# ...existing code...

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"<tweet_text>\"")
        sys.exit(1)
    tweet = sys.argv[1]
    result = predict_virality(tweet)
    
    print(f"Predicted Likes: {result['likes']}")
    print(f"Predicted Retweets: {result['retweets']}")
    print(f"Predicted Replies: {result['replies']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Virality Tier: {result['tier']}")
    print("\nFull result:")
    print(result)

