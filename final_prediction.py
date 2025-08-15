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

# Import new modules for topic and trending features (robust fallbacks)
try:
    from topic_category_hf import get_topic_category
except Exception:
    def get_topic_category(_):  # fallback to "general"/0
        return 0

try:
    from trending_hashtags import count_trending_hashtags
    TRENDING_OK = True
except Exception:
    TRENDING_OK = False
    def count_trending_hashtags(_):
        return 0

import re
NEWS_DOMAINS = ('bbc.', 'cnn.', 'nytimes.', 'reuters.', 'techcrunch.', 'washingtonpost.', 'theguardian.')
def _has_url(t: str) -> int:
    return 1 if re.search(r'https?://', t or '') else 0
def _has_news_url(t: str) -> int:
    if not t:
        return 0
    urls = re.findall(r'https?://[^\s]+', t)
    return 1 if any(any(nd in u for nd in NEWS_DOMAINS) for u in urls) else 0

# -----------------------------
# Default global blend weights
# -----------------------------
DEFAULT_GLOBAL_WEIGHTS = {
    "likes": 0.848,     # ML weight; Persona weight = 1 - this
    "retweets": 1.000,
    "replies": 0.000
}

BLEND_WEIGHTS_PATH = Path("blend_weights.json")

# --- Dynamic blending config (no DB; rule-based) ---
DYN_BOUNDS_DEFAULT = {
    "likes":    (0.55, 0.98),  # min,max ML weight
    "retweets": (0.80, 1.00),
    "replies":  (0.00, 0.60),
}

# --- Optional post-blend calibration (piecewise / isotonic arrays) ---
CALIBRATION_PATHS = {
    "likes":    Path("calibration_likes.json"),
    "retweets": Path("calibration_retweets.json"),
    "replies":  Path("calibration_replies.json"),
}

def _load_calibration(path: Path):
    """Expect a JSON like: {"x":[...], "y":[...], "log_space": true}."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        x = np.array(data.get("x", []), dtype=float)
        y = np.array(data.get("y", []), dtype=float)
        log_space = bool(data.get("log_space", True))
        if x.size and y.size and x.size == y.size:
            return {"x": x, "y": y, "log_space": log_space}
    except Exception as e:
        print(f"⚠️ Failed to load calibration from {path}: {e}")
    return None

CALIBRATORS = {k: _load_calibration(p) for k, p in CALIBRATION_PATHS.items()}

def _apply_calibration(metric: str, value: float) -> float:
    """Monotonic piecewise linear correction after blending."""
    cal = CALIBRATORS.get(metric)
    if not cal or value <= 0:
        return float(value)
    x, y, log_space = cal["x"], cal["y"], cal["log_space"]
    v = np.log1p(value) if log_space else float(value)
    v = np.clip(v, x.min(), x.max())
    corrected = float(np.interp(v, x, y))
    return float(np.expm1(corrected) if log_space else corrected)

def _ensemble_stats_new(model_list, features_dict):
    """For NEW heads: return p10/p50/p90 spread in original space."""
    if not model_list:
        return None
    m0 = model_list[0]
    names = getattr(m0, "feature_names_in_", None)
    names = list(names) if names is not None else get_new_model_features()
    X = pd.DataFrame([[features_dict.get(f, 0) for f in names]], columns=names)
    logs = np.array([float(m.predict(X)[0]) for m in model_list])
    vals = safe_expm1(np.clip(logs, None, 20.0))
    p10, p50, p90 = np.percentile(vals, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90), "mean": float(np.mean(vals))}

def _ensemble_stats_old(models, scaler, features_df):
    """For OLD heads: compute spread from per-model preds (denormalized)."""
    if not models:
        return None
    logs = np.array([float(m.predict(features_df)[0]) for m in models])
    denorm = safe_expm1(logs * scaler.get("std", 1.0) + scaler.get("mean", 0.0))
    p10, p50, p90 = np.percentile(denorm, [10, 50, 90])
    return {"p10": float(p10), "p50": float(p50), "p90": float(p90), "mean": float(np.mean(denorm))}

def _extract_cues(feats: dict):
    """Tiny, human-readable rule cues derived from text features."""
    tl = int(feats.get("text_length", 0))
    return {
        "qcta":      int(feats.get("question_word_count", 0) + feats.get("cta_word_count", 0) > 0),
        "trending":  int(feats.get("trending_hashtag_count", 0) > 0),
        "news":      int(feats.get("has_news_url", 0) == 1),
        "short":     int(tl > 0 and tl < 80),
        "long":      int(tl > 220),
        "zero_sent": int(abs(float(feats.get("sentiment_compound", 0.0))) < 1e-6),
    }

def _get_bounds(weights: dict, metric: str):
    lo = float(weights.get(f"{metric}_min_ml", DYN_BOUNDS_DEFAULT[metric][0]))
    hi = float(weights.get(f"{metric}_max_ml", DYN_BOUNDS_DEFAULT[metric][1]))
    lo = max(0.0, min(1.0, lo))
    hi = max(lo, min(1.0, hi))
    return (lo, hi)

def _dynamic_weight(metric: str, base_w: float, cues: dict, stats: dict | None, bounds: tuple[float, float]) -> float:
    """Rule-based dynamic ML weight per tweet; clamp to bounds."""
    w = float(base_w)

    # Content cues → trust ML more (esp. replies)
    if cues["qcta"] or cues["trending"] or cues["news"]:
        w += 0.08 if metric != "replies" else 0.15

    # Uncertainty: spread = (p90 - p10)/p50
    if stats and stats["p50"] > 0:
        spread = (stats["p90"] - stats["p10"]) / (stats["p50"] + 1e-6)
        if spread < 0.20:
            w += 0.07   # confident ML
        elif spread > 0.80:
            w -= 0.12   # uncertain ML → lean persona a bit

    # Off-distribution-ish heuristic
    if (cues["short"] or cues["long"]) and cues["zero_sent"]:
        w -= 0.05

    lo, hi = bounds
    return float(max(lo, min(hi, w)))

def _first_existing(*paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None

def load_blend_weights():
    """Load optional overrides from blend_weights.json; fall back to defaults.
    Supports extra keys:
      - force_static: bool
      - <metric>_min_ml / <metric>_max_ml, e.g. likes_min_ml: 0.6
    """
    weights = DEFAULT_GLOBAL_WEIGHTS.copy()
    # sensible defaults for bounds + mode flag
    weights["force_static"] = False
    for m in ("likes", "retweets", "replies"):
        weights[f"{m}_min_ml"] = DYN_BOUNDS_DEFAULT[m][0]
        weights[f"{m}_max_ml"] = DYN_BOUNDS_DEFAULT[m][1]

    if BLEND_WEIGHTS_PATH.exists():
        try:
            with open(BLEND_WEIGHTS_PATH, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            for k in ("likes", "retweets", "replies"):
                if k in overrides:
                    v = float(overrides[k])
                    weights[k] = max(0.0, min(1.0, v))
            # optional extras
            weights["force_static"] = bool(overrides.get("force_static", False))
            for m in ("likes", "retweets", "replies"):
                if f"{m}_min_ml" in overrides:
                    weights[f"{m}_min_ml"] = float(overrides[f"{m}_min_ml"])
                if f"{m}_max_ml" in overrides:
                    weights[f"{m}_max_ml"] = float(overrides[f"{m}_max_ml"])
        except Exception as e:
            print(f"⚠️ Failed to load blend_weights.json; using defaults. Error: {e}")
    return weights

# -----------------------------
# Model loading
# -----------------------------
print("Loading NEW retweet and reply models (if present)...")
# --- NEW: Likes ensemble/quantile (prefer *_final if present) ---
new_likes_ensemble = []
for seed in [42, 99, 123, 456, 789]:
    fn = _first_existing(f"NEW_likes_model_seed_{seed}_final.pkl",
                         f"NEW_likes_model_seed_{seed}.pkl")
    if fn:
        m = joblib.load(fn)
        new_likes_ensemble.append(m)
        if len(new_likes_ensemble) == 1 and hasattr(m, "feature_names_in_"):
            print(f"✅ NEW Likes Model Features: {list(m.feature_names_in_)}")
    else:
        print(f"⚠️ NEW_likes model for seed {seed} not found")

qpath = _first_existing("NEW_likes_quantile_model_final.pkl",
                        "NEW_likes_quantile_model.pkl")
new_likes_quantile = joblib.load(qpath) if qpath else None
if new_likes_quantile is None:
    print("⚠️ NEW_likes_quantile_model not found")

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
    feats = extract_features_from_text(tweet_text)
    text_lower = (tweet_text or "").lower()
    # Core NEW features
    feats.setdefault("text_length", len(tweet_text or ""))
    feats.setdefault("has_media", 0)   # unknown at inference => 0
    feats.setdefault("hour", 12)       # override from caller if you know it
    feats["has_url"] = _has_url(tweet_text)
    feats["has_news_url"] = _has_news_url(tweet_text)
    # Trending hashtags count (safe fallback if module missing)
    feats["trending_hashtag_count"] = count_trending_hashtags(tweet_text) if TRENDING_OK else 0
    # Topic id/category (fallback defined in import)
    feats["topic_category"] = get_topic_category(tweet_text)
    # Reply cues
    feats.setdefault("question_word_count",
        sum(1 for w in ["what","how","why","when","where","who","which"] if w in text_lower))
    feats.setdefault("cta_word_count",
        sum(1 for w in ["please","help","share","comment","reply","thoughts","rt"] if w in text_lower))
    # Sentiment/emotions fallbacks
    feats.setdefault("sentiment_compound", 0.0)
    feats.setdefault("sentiment_subjectivity", 0.5)
    for emo in ["anger","fear","joy","sadness","surprise"]:
        feats.setdefault(f"emotion_{emo}", 0.0)
    return feats

def predict_metric_ml_old(models, scaler, features_df):
    """Old models: average ensemble, unscale from log space.
    Returns float (keep precision; no rounding/casting)."""
    if not models:
        return 0.0
    preds = [float(m.predict(features_df)[0]) for m in models]
    base = float(np.mean(preds))
    pred_log = base * scaler.get("std", 1.0) + scaler.get("mean", 0.0)
    return float(max(0.0, safe_expm1(pred_log)))


def _predict_new_core(model_or_list, features_dict):
    """Run NEW model(s) in log space and return float on original scale (no rounding)."""
    if isinstance(model_or_list, list) and model_or_list:
        m0 = model_or_list[0]
        feature_names = getattr(m0, "feature_names_in_", None)
        names = list(feature_names) if feature_names is not None else get_new_model_features()
        X = pd.DataFrame([[features_dict.get(f, 0) for f in names]], columns=names)
        log_pred = float(np.mean([float(m.predict(X)[0]) for m in model_or_list]))
    elif model_or_list is not None:
        feature_names = getattr(model_or_list, "feature_names_in_", None)
        names = list(feature_names) if feature_names is not None else get_new_model_features()
        X = pd.DataFrame([[features_dict.get(f, 0) for f in names]], columns=names)
        log_pred = float(model_or_list.predict(X)[0])
    else:
        return 0.0
    return float(min(safe_expm1(log_pred), 100000.0))


# --- NEW: helpers to get ensemble/old-model percentiles on ORIGINAL scale ---
def _ensemble_preds(models_or_list, features_dict) -> list[float]:
    """Return list of per-model predictions on original (expm1) scale."""
    if not models_or_list:
        return []
    models = models_or_list if isinstance(models_or_list, list) else [models_or_list]
    m0 = models[0]
    feature_names = getattr(m0, "feature_names_in_", None)
    names = list(feature_names) if feature_names is not None else get_new_model_features()
    X = pd.DataFrame([[features_dict.get(f, 0) for f in names]], columns=names)
    logs = [float(m.predict(X)[0]) for m in models]
    vals = [float(min(safe_expm1(lp), 100000.0)) for lp in logs]
    return vals

def _old_ensemble_preds(models, features_df) -> list[float]:
    """Return list of per-model predictions on original scale using OLD models & scalers are applied via existing helper."""
    if not models:
        return []
    # NOTE: OLD models were trained in log space; we reuse your existing predict path per model
    preds = []
    for m in models:
        # emulate your old averaging path but per-model
        base_log = float(m.predict(features_df)[0])
        preds.append(float(max(0.0, safe_expm1(base_log))) )  # unscaled single-model view
    return preds

def _pct_dict(values: list[float]) -> dict:
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }

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

    # Likes
    if new_likes_ensemble:
        print("🚀 Using NEW Likes Ensemble Models")
        ml_likes = _predict_new_core(new_likes_ensemble, enhanced)
    elif new_likes_quantile:
        print("🚀 Using NEW Likes Quantile Model")
        ml_likes = _predict_new_core(new_likes_quantile, enhanced)
    else:
        print("⚠️ Using OLD Likes Models")
        ml_likes = predict_metric_ml_old(reg_models_likes, likes_scaler, features_df)

    # --- ML distribution stats for dynamic weighting (if ensembles present) ---
    likes_stats = (_ensemble_stats_new(new_likes_ensemble, enhanced)
                   if new_likes_ensemble else
                   _ensemble_stats_old(reg_models_likes, likes_scaler, features_df))
    rt_stats    = (_ensemble_stats_new(new_rt_ensemble, enhanced)
                   if new_rt_ensemble else
                   _ensemble_stats_old(reg_models_rt, rt_scaler, features_df))
    reply_stats = (_ensemble_stats_new(new_reply_ensemble, enhanced)
                   if new_reply_ensemble else
                   _ensemble_stats_old(reg_models_reply, reply_scaler, features_df))

    cues = _extract_cues(enhanced)

    # Persona (as float)
    persona = aggregate_engagement(tweet_text, PERSONAS)
    persona_likes = float(persona.get("persona_likes", 0.0))
    persona_retweets = float(persona.get("persona_rts", 0.0))
    persona_replies = float(persona.get("persona_replies", 0.0))

    # Global OR Dynamic blending (respects force_static + per-metric bounds)
    if weights.get("force_static", False):
        w_likes   = float(weights.get("likes",    DEFAULT_GLOBAL_WEIGHTS["likes"]))
        w_rts     = float(weights.get("retweets", DEFAULT_GLOBAL_WEIGHTS["retweets"]))
        w_replies = float(weights.get("replies",  DEFAULT_GLOBAL_WEIGHTS["replies"]))
        weights_mode = "static"
    else:
        w_likes = _dynamic_weight(
            "likes",
            float(weights.get("likes", DEFAULT_GLOBAL_WEIGHTS["likes"])),
            cues, likes_stats, _get_bounds(weights, "likes")
        )
        w_rts = _dynamic_weight(
            "retweets",
            float(weights.get("retweets", DEFAULT_GLOBAL_WEIGHTS["retweets"])),
            cues, rt_stats, _get_bounds(weights, "retweets")
        )
        w_replies = _dynamic_weight(
            "replies",
            float(weights.get("replies", DEFAULT_GLOBAL_WEIGHTS["replies"])),
            cues, reply_stats, _get_bounds(weights, "replies")
        )
        weights_mode = "dynamic"

    blended_likes    = (ml_likes    * w_likes)   + (persona_likes    * (1.0 - w_likes))
    blended_retweets = (ml_retweets * w_rts)     + (persona_retweets * (1.0 - w_rts))
    blended_replies  = (ml_replies  * w_replies) + (persona_replies  * (1.0 - w_replies))

    # --- NEW: expose ML percentile spreads for uncertainty-aware ranges ---
    ml_percentiles = {}

    # Likes
    if new_likes_ensemble:
        preds = _ensemble_preds(new_likes_ensemble, enhanced)
        ml_percentiles["likes"] = _pct_dict(preds)
    elif reg_models_likes:
        # approximate spread from OLD models if NEW absent
        preds = _old_ensemble_preds(reg_models_likes, features_df)
        ml_percentiles["likes"] = _pct_dict(preds)

    # Retweets
    if new_rt_ensemble:
        preds = _ensemble_preds(new_rt_ensemble, enhanced)
        ml_percentiles["retweets"] = _pct_dict(preds)
    elif reg_models_rt:
        preds = _old_ensemble_preds(reg_models_rt, features_df)
        ml_percentiles["retweets"] = _pct_dict(preds)

    # Replies
    if new_reply_ensemble:
        preds = _ensemble_preds(new_reply_ensemble, enhanced)
        ml_percentiles["replies"] = _pct_dict(preds)
    elif reg_models_reply:
        preds = _old_ensemble_preds(reg_models_reply, features_df)
        ml_percentiles["replies"] = _pct_dict(preds)

    # Post-blend calibration (no-op if files are absent)
    blended_likes    = _apply_calibration("likes", blended_likes)
    blended_retweets = _apply_calibration("retweets", blended_retweets)
    blended_replies  = _apply_calibration("replies", blended_replies)

    return {
        "weights_used": {"likes": w_likes, "retweets": w_rts, "replies": w_replies},
        "weights_mode": weights_mode,
        "models_used": {
            "likes":    "NEW" if (new_likes_ensemble or new_likes_quantile) else "OLD",
            "retweets": "NEW" if (new_rt_ensemble    or new_rt_quantile)    else "OLD",
            "replies":  "NEW" if (new_reply_ensemble or new_reply_quantile) else "OLD",
        },
        "calibration": {k: bool(CALIBRATORS.get(k)) for k in ("likes","retweets","replies")},
        "ml":      {"likes": ml_likes, "retweets": ml_retweets, "replies": ml_replies},
        "persona": {"likes": persona_likes, "retweets": persona_retweets, "replies": persona_replies},
        "blended": {"likes": blended_likes, "retweets": blended_retweets, "replies": blended_replies},

        # NEW: percentiles for uncertainty-aware ranges
        "ml_percentiles": ml_percentiles
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
        print(f"{k.capitalize():8s}: {int(round(v))}")

    print("\n=== PERSONA ENGINE PREDICTIONS ===")
    for k, v in result["persona"].items():
        print(f"{k.capitalize():8s}: {int(round(v))}")

    print("\n=== BLENDED PREDICTIONS ===")
    for k, v in result["blended"].items():
        print(f"{k.capitalize():8s}: {int(round(v))}")
