from features import extract_features_from_text, select_features, MODEL_FEATURES
from persona_engine import load_personas, aggregate_engagement
from calibrator import ViralCalibrator
import joblib
import glob
import os

# Initialize once
PERSONAS = load_personas("personas.json")

# Load all ensemble regression models from the new folder
MODEL_PATHS = [
    os.path.join(os.path.dirname(__file__), f"regression_xgb_seed_{seed}.pkl")
    for seed in [42, 99, 123, 456, 789]
]
REGRESSION_MODELS = [joblib.load(path) for path in MODEL_PATHS if os.path.exists(path)]
if not REGRESSION_MODELS:
    raise RuntimeError("No ensemble regression models found in 'new' folder.")

# Use the official MODEL_FEATURES from features.py (27 features in correct order)
TRAINING_FEATURES = MODEL_FEATURES

CALIBRATOR = ViralCalibrator("ready_datazet.csv")

def extract_features(tweet_text):
    return extract_features_from_text(tweet_text)

def detect_viral_signals(tweet_text):
    signals = []
    if "?" in tweet_text:
        signals.append("Question hook")
    if "\n\n" in tweet_text:
        signals.append("Multi-paragraph structure")
    if any(word in tweet_text.lower() for word in ["learn", "secret", "method"]):
        signals.append("Curiosity trigger")
    if sum(c.isdigit() for c in tweet_text) > 2:
        signals.append("Number credibility")
    return signals


def predict_virality(tweet_text):
    # 1. Extract features for ML models (27 features)
    features = extract_features(tweet_text)
    
    # 2. ML prediction - Filter to only MODEL_FEATURES in correct order
    import pandas as pd
    features_df = pd.DataFrame([features])
    features_df = features_df[MODEL_FEATURES]  # Select only 27 model features, in order
    
    # 3. ML ensemble prediction (average output)
    model_preds = [model.predict(features_df)[0] for model in REGRESSION_MODELS]
    avg_pred = sum(model_preds) / len(model_preds)
    std_pred = (sum((p - avg_pred) ** 2 for p in model_preds) / len(model_preds)) ** 0.5
    ml_pred = {'likes': avg_pred, 'std': std_pred}
    
    # 4. Persona simulation (independent - no ML features needed)
    persona_result = aggregate_engagement(tweet_text, PERSONAS)  # Just passes tweet text
    persona_likes = persona_result['persona_likes']
    
    # 5. Calibrate - prepare features for calibrator (legacy compatibility)
    # Extract individual features from the features dictionary for calibrator
    features_dict = features if isinstance(features, dict) else features.iloc[0].to_dict()
    
    calibrator_features = {
        'question_count': features_dict.get('question_count', 0),
        'exclamation_count': features_dict.get('exclamation_count', 0), 
        'paragraph_count': features_dict.get('paragraph_count', 0),
        'sentiment_compound': features_dict.get('sentiment_compound', 0),
        'noun_count': features_dict.get('noun_count', 0),
        'engagement_score': features_dict.get('engagement_score', 0),
        'char_count': features_dict.get('char_count', 0),
        'hashtag_count': features_dict.get('hashtag_count', 0),
        'mention_count': features_dict.get('mention_count', 0)
    }
    
    # Suppress sklearn warnings for cleaner output
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = CALIBRATOR.predict(calibrator_features)

    # Three-tier prediction
    if result['virality_tier'] == 'High':
        final_pred = result['predicted_likes'] * 1.5
    else:
        final_pred = result['predicted_likes']

    # Confidence estimation
    confidence = f"{int(result['virality_prob'] * 100)}%"

    return {
        'predicted_likes': int(final_pred),
        'confidence': confidence,
        'ml_prediction': ml_pred['likes'],
        'ml_std': ml_pred['std'],
        'persona_prediction': persona_likes,
        'top_demographics': persona_result['top_segments'][:3],
        'viral_signals': detect_viral_signals(tweet_text),
        'tier': result['virality_tier']
    }

# Add after real engagement data comes in

def update_calibrator(tweet_text, actual_likes):
    # Extract features for ML models (27 features)
    features = extract_features(tweet_text)
    
    # Filter to only MODEL_FEATURES in correct order
    import pandas as pd
    features_df = pd.DataFrame([features])
    features_df = features_df[MODEL_FEATURES]  # Select only 27 model features, in order
    
    model_preds = [model.predict(features_df)[0] for model in REGRESSION_MODELS]
    avg_pred = sum(model_preds) / len(model_preds)
    
    # Persona simulation (independent - no ML features needed)
    persona_pred = aggregate_engagement(tweet_text, PERSONAS)['persona_likes']  # Just passes tweet text
    
    CALIBRATOR.add_data_point(avg_pred, persona_pred, actual_likes)
    CALIBRATOR.calibrate()

# Command-line interface
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict_unified.py \"<tweet_text>\"")
        sys.exit(1)
    
    tweet = sys.argv[1]
    try:
        result = predict_virality(tweet)
        print("\n🎯 VIRALITY PREDICTION RESULTS:")
        print(f"📊 Predicted Likes: {result['predicted_likes']:,}")
        print(f"🎯 Confidence: {result['confidence']}")
        print(f"🏆 Tier: {result['tier']}")
        print(f"🤖 ML Prediction: {result['ml_prediction']:.0f}")
        print(f"👥 Persona Prediction: {result['persona_prediction']:.0f}")
        print(f"🎪 Top Demographics: {', '.join(result['top_demographics'])}")
        if result['viral_signals']:
            print(f"🚀 Viral Signals: {', '.join(result['viral_signals'])}")
        print(f"\n📋 Full Result: {result}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
