# final_prediction.py
# Blended predictions using global weights from weightage.py (leakage-free).
# Likes:   ML 0.848 / Persona 0.152
# Retweets: ML 1.000 / Persona 0.000
# Replies:  ML 0.000 / Persona 1.000
#
# Optional override file: blend_weights.json
#   Example: {"likes": 0.83, "retweets": 1.0, "replies": 0.05}

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from features import extract_features_from_text, MODEL_FEATURES
from persona_engine import load_personas, aggregate_engagement

# Import new modules for topic and trending features
from topic_category_hf import get_topic_category
from trending_hashtags import count_trending_hashtags

# -----------------------------
# Default global blend weights
# -----------------------------
DEFAULT_GLOBAL_WEIGHTS = {
    "likes": 0.848,     # ML weight; Persona weight = 1 - this
    "retweets": 1.000,
    "replies": 0.000
}

BLEND_WEIGHTS_PATH = Path("blend_weights.json")

def load_blend_weights():
    """Load optional overrides from blend_weights.json; fall back to defaults."""
    weights = DEFAULT_GLOBAL_WEIGHTS.copy()
    if BLEND_WEIGHTS_PATH.exists():
        try:
            with open(BLEND_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            for k in ["likes", "retweets", "replies"]:
                if k in overrides:
                    v = float(overrides[k])
                    weights[k] = max(0.0, min(1.0, v))  # clip to [0,1]
        except Exception as e:
            print(f"⚠️ Failed to load blend_weights.json; using defaults. Error: {e}")
    return weights

# -----------------------------
# Model loading
# -----------------------------
print("Loading NEW retweet and reply models (if present)...")

# NEW Retweet ensemble/quantile
new_rt_ensemble = []
for seed in [42, 99, 123, 456, 789]:
    fn = f"NEW_retweet_model_seed_{seed}.pkl"
    if Path(fn).exists():
        m = joblib.load(fn)
        new_rt_ensemble.append(m)
        if len(new_rt_ensemble) == 1 and hasattr(m, "feature_names_in_"):
            print(f"✅ NEW Retweet Model Features: {list(m.feature_names_in_)}")
    else:
        print(f"⚠️ {fn} not found")
new_rt_quantile = joblib.load("NEW_retweet_quantile_model.pkl") if Path("NEW_retweet_quantile_model.pkl").exists() else None
if new_rt_quantile is None:
    print("⚠️ NEW_retweet_quantile_model.pkl not found")

# NEW Reply ensemble/quantile
new_reply_ensemble = []
for seed in [42, 99, 123, 456, 789]:
    fn = f"NEW_reply_model_seed_{seed}.pkl"
    if Path(fn).exists():
        m = joblib.load(fn)
        new_reply_ensemble.append(m)
        if len(new_reply_ensemble) == 1 and hasattr(m, "feature_names_in_"):
            print(f"✅ NEW Reply Model Features: {list(m.feature_names_in_)}")
    else:
        print(f"⚠️ {fn} not found")
new_reply_quantile = joblib.load("NEW_reply_quantile_model.pkl") if Path("NEW_reply_quantile_model.pkl").exists() else None
if new_reply_quantile is None:
    print("⚠️ NEW_reply_quantile_model.pkl not found")

# OLD fallbacks (likes/rt/replies)
def _try_load_old(path_fmt, seeds):
    models = []
    for s in seeds:
        fn = path_fmt.format(s)
        if Path(fn).exists():
            models.append(joblib.load(fn))
    return models

reg_seeds = [42, 99, 123, 456, 789]

reg_models_likes = _try_load_old("regression_xgb_seed_{}.pkl", reg_seeds)
likes_scaler = joblib.load("likes_log_scaler.pkl") if Path("likes_log_scaler.pkl").exists() else {"mean": 0.0, "std": 1.0}
print("✅ Loaded OLD likes models" if reg_models_likes else "⚠️ OLD likes models not found")

reg_models_rt = _try_load_old("regression_rt_xgb_seed_{}.pkl", reg_seeds)
rt_scaler = joblib.load("retweets_log_scaler.pkl") if Path("retweets_log_scaler.pkl").exists() else {"mean": 0.0, "std": 1.0}
print("✅ Loaded OLD retweet models" if reg_models_rt else "⚠️ OLD retweet models not found")

reg_models_reply = _try_load_old("regression_reply_xgb_seed_{}.pkl", reg_seeds)
reply_scaler = joblib.load("replies_log_scaler.pkl") if Path("replies_log_scaler.pkl").exists() else {"mean": 0.0, "std": 1.0}
print("✅ Loaded OLD reply models" if reg_models_reply else "⚠️ OLD reply models not found")

# Personas
PERSONAS = load_personas("personas.json")

# -----------------------------
# Utilities
# -----------------------------
def safe_expm1(x):
    # guard against overflow; exp(20) ~ 4.85e8
    return np.expm1(np.clip(x, None, 20.0))

def get_new_model_features():
    """Feature list NEW models expect (keep in sync with training)."""
    base = [
        "char_count", "word_count", "sentence_count", "hashtag_count",
        "mention_count", "question_count", "exclamation_count",
        "day_of_week", "is_weekend"
    ]
    new = [
        "text_length", "has_media", "hour", "has_url", "has_news_url",
        "trending_hashtag_count",      # NOTE: fixed spelling
        "topic_category"
    ]
    sentiment = ["sentiment_compound", "sentiment_subjectivity"]
    emotions = ["emotion_anger", "emotion_fear", "emotion_joy", "emotion_sadness", "emotion_surprise"]
    reply_only = ["question_word_count", "cta_word_count"]
    return base + new + sentiment + emotions + reply_only

def prepare_features_for_new_models(tweet_text: str) -> dict:
    """Build a dict with every feature NEW models might ask for."""
    feats = extract_features_from_text(tweet_text)  # your base extractor
    text_lower = tweet_text.lower()

    # Populate missing NEW features
    feats.setdefault("text_length", len(tweet_text))
    feats.setdefault("has_media", 0)  # update upstream if media is known
    feats.setdefault("hour", 12)      # can be overridden by caller if desired
    feats.setdefault("has_url", 1 if ("http://" in text_lower or "https://" in text_lower) else 0)
    feats.setdefault("has_news_url", 0)
    
    # Use new trending hashtag module
    feats.setdefault("trending_hashtag_count", count_trending_hashtags(tweet_text))

    # Use new topic category module
    feats["topic_category"] = get_topic_category(tweet_text)

    # Reply cues
    feats.setdefault("question_word_count", sum(1 for w in ["what","how","why","when","where","who","which"] if w in text_lower))
    feats.setdefault("cta_word_count", sum(1 for w in ["please","help","share","comment","reply","thoughts"] if w in text_lower))

    # Sentiment/emotions fallbacks
    feats.setdefault("sentiment_compound", 0.0)
    feats.setdefault("sentiment_subjectivity", 0.5)
    for emo in ["anger","fear","joy","sadness","surprise"]:
        feats.setdefault(f"emotion_{emo}", 0.0)

    return feats

def predict_metric_ml_old(models, scaler, features_df):
    """Old models: average ensemble, unscale from log space."""
    if not models:
        return 0
    preds = [m.predict(features_df)[0] for m in models]
    base = float(np.mean(preds))
    pred_log = base * scaler.get("std", 1.0) + scaler.get("mean", 0.0)
    return int(safe_expm1(pred_log))

def _predict_new_core(model_or_list, features_dict):
    """Helper: run NEW model(s) in log space and return int on original scale."""
    if isinstance(model_or_list, list) and model_or_list:
        m0 = model_or_list[0]
        feature_names = getattr(m0, "feature_names_in_", None)
        if feature_names is not None:
            names = list(feature_names)
        else:
            names = get_new_model_features()
        X = pd.DataFrame([[features_dict.get(f, 0) for f in names]], columns=names)
        log_pred = float(np.mean([m.predict(X)[0] for m in model_or_list]))
    elif model_or_list is not None:
        feature_names = getattr(model_or_list, "feature_names_in_", None)
        if feature_names is not None:
            names = list(feature_names)
        else:
            names = get_new_model_features()
        X = pd.DataFrame([[features_dict.get(f, 0) for f in names]], columns=names)
        log_pred = float(model_or_list.predict(X)[0])
    else:
        return 0
    return int(min(safe_expm1(log_pred), 100000))

# -----------------------------
# Main blend function
# -----------------------------
def predict_blended(tweet_text: str, blend_weights: dict | None = None):
    """Return ML, Persona, and blended predictions using global weights per metric."""
    weights = load_blend_weights() if blend_weights is None else blend_weights

    # Prepare features for both NEW and OLD models
    enhanced = prepare_features_for_new_models(tweet_text)
    base = extract_features_from_text(tweet_text)

    # Ensure OLD feature order
    try:
        features_df = pd.DataFrame([base])[MODEL_FEATURES]
    except Exception:
        cols = list(MODEL_FEATURES)
        features_df = pd.DataFrame([[base.get(c, 0) for c in cols]], columns=cols)

    # --- ML predictions ---
    # Retweets
    if new_rt_ensemble:
        print("🚀 Using NEW Retweet Ensemble Models")
        ml_retweets = _predict_new_core(new_rt_ensemble, enhanced)
    elif new_rt_quantile:
        print("🚀 Using NEW Retweet Quantile Model")
        ml_retweets = _predict_new_core(new_rt_quantile, enhanced)
    else:
        print("⚠️ Using OLD Retweet Models")
        ml_retweets = predict_metric_ml_old(reg_models_rt, rt_scaler, features_df)

    # Replies
    if new_reply_ensemble:
        print("🚀 Using NEW Reply Ensemble Models")
        ml_replies = _predict_new_core(new_reply_ensemble, enhanced)
    elif new_reply_quantile:
        print("🚀 Using NEW Reply Quantile Model")
        ml_replies = _predict_new_core(new_reply_quantile, enhanced)
    else:
        print("⚠️ Using OLD Reply Models")
        ml_replies = predict_metric_ml_old(reg_models_reply, reply_scaler, features_df)

    # Likes (still OLD)
    print("⚠️ Using OLD Likes Models")
    ml_likes = predict_metric_ml_old(reg_models_likes, likes_scaler, features_df)

    # --- Persona predictions ---
    persona = aggregate_engagement(tweet_text, PERSONAS)
    persona_likes = int(persona.get("persona_likes", 0))
    persona_retweets = int(persona.get("persona_rts", 0))
    persona_replies = int(persona.get("persona_replies", 0))

    # --- Global blending (no leakage) ---
    w_likes = float(weights.get("likes", DEFAULT_GLOBAL_WEIGHTS["likes"]))
    w_rts = float(weights.get("retweets", DEFAULT_GLOBAL_WEIGHTS["retweets"]))
    w_replies = float(weights.get("replies", DEFAULT_GLOBAL_WEIGHTS["replies"]))

    blended_likes = int(round(ml_likes * w_likes + persona_likes * (1.0 - w_likes)))
    blended_retweets = int(round(ml_retweets * w_rts + persona_retweets * (1.0 - w_rts)))
    blended_replies = int(round(ml_replies * w_replies + persona_replies * (1.0 - w_replies)))

    return {
        "weights_used": {"likes": w_likes, "retweets": w_rts, "replies": w_replies},
        "models_used": {
            "likes": "OLD",
            "retweets": "NEW" if (new_rt_ensemble or new_rt_quantile) else "OLD",
            "replies": "NEW" if (new_reply_ensemble or new_reply_quantile) else "OLD",
        },
        "ml": {"likes": ml_likes, "retweets": ml_retweets, "replies": ml_replies},
        "persona": {"likes": persona_likes, "retweets": persona_retweets, "replies": persona_replies},
        "blended": {"likes": blended_likes, "retweets": blended_retweets, "replies": blended_replies},
    }

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python final_prediction.py "your tweet text here"')
        sys.exit(1)

    tweet = sys.argv[1]
    result = predict_blended(tweet)

    print("\n=== BLEND WEIGHTS (Global, leakage-free) ===")
    print(f"Likes:    ML {result['weights_used']['likes']:.3f} | Persona {1 - result['weights_used']['likes']:.3f}")
    print(f"Retweets: ML {result['weights_used']['retweets']:.3f} | Persona {1 - result['weights_used']['retweets']:.3f}")
    print(f"Replies:  ML {result['weights_used']['replies']:.3f} | Persona {1 - result['weights_used']['replies']:.3f}")

    print("\n=== MODELS USED ===")
    for k, v in result["models_used"].items():
        print(f"{k.capitalize():8s}: {v}")

    print("\n=== ML PREDICTIONS ===")
    for k, v in result["ml"].items():
        print(f"{k.capitalize():8s}: {v}")

    print("\n=== PERSONA ENGINE PREDICTIONS ===")
    for k, v in result["persona"].items():
        print(f"{k.capitalize():8s}: {v}")

    print("\n=== BLENDED PREDICTIONS ===")
    for k, v in result["blended"].items():
        print(f"{k.capitalize():8s}: {v}")
