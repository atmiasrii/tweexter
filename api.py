# api.py
# FastAPI wrapper for Tweexter predictions with follower-aware scaling.

import os           # NEW
import csv          # NEW
import time         # NEW
from datetime import datetime  # NEW
import re  # NEW
import warnings
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel, Field, validator
from starlette.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# --- timing log setup ---
from pathlib import Path
LOG_PATH = Path("predict_timing.csv")

def _log_timing_row(row: dict) -> None:
    """Append a single timing row to predict_timing.csv with a stable header."""
    cols = [
        "ts",
        "elapsed_ms",
        "followers",
        "text_len",
        "likes_mid",
        "retweets_mid",
        "replies_mid",
        "weights_mode",
        "models_likes",
        "models_retweets",
        "models_replies",
    ]
    # write header once
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        # keep only known keys; missing keys become empty
        safe = {k: row.get(k, "") for k in cols}
        w.writerow(safe)

# Suppress version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", message=".*XGBoost.*")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

# Import the base blended predictor
from final_prediction import predict_blended
from features import extract_features_from_text  # NEW

# Import follower-scaling helpers from your final.py
from final import (
    load_cfg,
    apply_follower_scaling,
    pick_blend_weights,
    factor_for,
    baselines_for,
    baseline_weight,
    reply_like_ratio_for,  # NEW
)

app = FastAPI(title="Tweexter API", version="1.0.0")

# CORS (wide open for local dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # <- wide open for dev
    allow_credentials=False,      # keep False when using "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------

class PredictRequest(BaseModel):
    text: str = Field(..., description="Tweet text to predict on", max_length=2000)
    # Make followers optional with a sensible default so callers can send only {"text": "..."}.
    # Assumption: default to 100 followers when not provided.
    followers: int = Field(100, ge=1, description="Author follower count (defaults to 100)")
    return_details: bool = Field(False, description="Include full breakdown")

    @validator("text")
    def non_empty_text(cls, v: str):
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


# NEW: a small schema for per-metric ranges
class MetricRange(BaseModel):
    low: int
    mid: int
    high: int


class PredictResponse(BaseModel):
    likes: int
    retweets: int
    replies: int
    ranges: Dict[str, MetricRange]  # NEW: per-metric ranges (low/mid/high)
    details: Optional[Dict[str, Any]] = None


# NEW: follower-tier → default relative band (step 2)
TIER_RANGE_BANDS = [
    (1000,     0.50),  # ≤1K: ±50%
    (5000,     0.35),  # ≤5K: ±35%
    (10000,    0.30),  # ≤10K: ±30%
    (50000,    0.25),  # ≤50K: ±25%
    (100000,   0.22),  # ≤100K: ±22%
    (300000,   0.20),  # ≤300K: ±20%
    (600000,   0.18),  # ≤600K: ±18%
    (1000000,  0.16),  # ≤1M: ±16%
    (None,     0.15),  # >1M: ±15%
]

def range_band_for_followers(followers: int) -> float:
    """Return the relative ±band for a given follower count."""
    f = int(max(1, followers))
    for max_followers, band in TIER_RANGE_BANDS:
        if max_followers is None or f <= max_followers:
            return float(band)
    return 0.20  # safe fallback


# -------- Metric-specific band scaling (step 3) --------
LIKE_BAND_FLOOR = 8      # min ±likes
RT_BAND_MIN_ABS = 5      # min ±retweets
REPLY_BAND_MIN_ABS = 2   # min ±replies
RETWEET_BAND_SCALE = 0.70
REPLY_BAND_SCALE   = 0.50

# -------- Content-cue heuristics (step 5) --------
NEWS_DOMAINS = ('bbc.', 'cnn.', 'nytimes.', 'reuters.', 'techcrunch.', 'washingtonpost.', 'theguardian.')

def _has_url(t: str) -> int:
    return 1 if re.search(r'https?://', t or '') else 0

def _has_news_url(t: str) -> int:
    if not t:
        return 0
    urls = re.findall(r'https?://[^\s]+', t)
    return 1 if any(any(nd in u for nd in NEWS_DOMAINS) for u in urls) else 0

def _content_cues(text: str) -> dict:
    """Lightweight cues we'll use to shape the UPPER bounds."""
    feats = extract_features_from_text(text or "")
    text_has_url = _has_url(text or "")
    text_has_news = _has_news_url(text or "")

    trending_ct = int(feats.get("trending_hashtag_count", 0))
    q_ct = int(feats.get("question_count", 0))
    q_words = int(feats.get("question_word_count", 0))
    cta = int(feats.get("cta_word_count", 0))

    char_count = int(feats.get("char_count", len(text or "")))
    hashtag_count = int(feats.get("hashtag_count", 0))
    mention_count = int(feats.get("mention_count", 0))
    sent = float(feats.get("sentiment_compound", 0.0))

    # short, punchy, positive one-liner heuristic
    short_positive = (
        char_count <= 120 and sent >= 0.35 and hashtag_count == 0 and mention_count <= 1 and text_has_url == 0
    )

    return {
        "has_url": text_has_url,
        "has_news_url": text_has_news,
        "trending": 1 if trending_ct > 0 else 0,
        "qa_cta": 1 if (q_ct + q_words + cta) > 0 else 0,
        "short_positive": 1 if short_positive else 0,
    }

def _blend_with_uncertainty(metric: str, mid: int, rel_band: float, floors: dict, base_result: dict) -> float:
    """
    Step 4: If ML P10/P90 available, widen band using the wider of (tier band vs data band).
    Supports base_result['ml_percentiles'] or base_result['ml_spread'] structure if present.
    Returns a RELATIVE band to use for this metric.
    """
    used_rel = float(rel_band)

    # Try ml_percentiles
    mp = base_result.get("ml_percentiles", {})
    if isinstance(mp, dict) and metric in mp and all(k in mp[metric] for k in ("p10", "p90")):
        p10 = float(mp[metric]["p10"])
        p90 = float(mp[metric]["p90"])
        if mid > 0:
            data_rel = max(abs(mid - p10), abs(p90 - mid)) / float(mid)
            used_rel = max(used_rel, data_rel)

    # Try ml_spread fallback
    ms = base_result.get("ml_spread", {})
    if isinstance(ms, dict) and metric in ms and all(k in ms[metric] for k in ("p10", "p90")):
        p10 = float(ms[metric]["p10"])
        p90 = float(ms[metric]["p90"])
        if mid > 0:
            data_rel = max(abs(mid - p10), abs(p90 - mid)) / float(mid)
            used_rel = max(used_rel, data_rel)

    # Keep a sane minimum absolute floor (relative band might be tiny on small mids)
    # Floors are enforced later when we compute absolute deltas.
    return used_rel

def _make_range_from_band(mid: int, rel_band: float, min_abs: int) -> tuple[int, int]:
    """Return (low, high) given a midpoint, relative band, and absolute minimum delta."""
    delta = max(int(round(abs(mid) * rel_band)), int(min_abs))
    low = max(0, mid - delta)
    high = mid + delta
    return low, high

def _apply_content_cues_to_highs(likes_high: int, rt_high: int, reply_high: int,
                                 likes_mid: int, rt_mid: int, reply_mid: int,
                                 cues: dict, cue_bumps: dict = None) -> tuple[int, int, int]:
    """
    Step 5: bump or trim ONLY upper bounds based on content cues.
    Never push an upper bound below its mid.
    """
    if cue_bumps is None:
        cue_bumps = {}
    
    lh, rh, ph = int(likes_high), int(rt_high), int(reply_high)

    # Upside: trending hashtags / news link
    if cues.get("trending", 0) == 1:
        trending_bump = 1.0 + float(cue_bumps.get("trending", 0.15))
        lh = int(round(lh * trending_bump))
        rh = int(round(rh * trending_bump))
    if cues.get("has_news_url", 0) == 1:
        news_bump = 1.0 + float(cue_bumps.get("news", 0.10))
        lh = int(round(lh * news_bump))
        rh = int(round(rh * news_bump))

    # Upside: short, punchy, positive one-liner
    if cues.get("short_positive", 0) == 1:
        short_pos_bump = 1.0 + float(cue_bumps.get("short_positive", 0.10))
        lh = int(round(lh * short_pos_bump))
        rh = int(round(rh * short_pos_bump))

    # Upside: question / CTA → replies jump; small nudge for likes
    if cues.get("qa_cta", 0) == 1:
        qa_reply_bump = 1.0 + float(cue_bumps.get("qa_cta_reply", 0.25))
        qa_like_bump = 1.0 + float(cue_bumps.get("qa_cta_like", 0.05))
        ph = int(round(ph * qa_reply_bump))
        lh = int(round(lh * qa_like_bump))

    # Downside: external non-news link → shave upper a bit
    if cues.get("has_url", 0) == 1 and cues.get("has_news_url", 0) == 0:
        nonnews_bump = 1.0 + float(cue_bumps.get("nonnews_link", -0.10))
        lh = int(round(lh * nonnews_bump))
        rh = int(round(rh * nonnews_bump))

    # never below mid
    lh = max(lh, int(likes_mid))
    rh = max(rh, int(rt_mid))
    ph = max(ph, int(reply_mid))
    return lh, rh, ph


# -------- Viral ceiling (step 6) --------
def _viral_ceiling_factors(cues: dict, followers: int, caps: dict = None) -> dict:
    """
    Return max multipliers for uppers when strong viral triggers exist.
    Likes/RTs: up to ~3x mid; Replies: up to ~2x mid (more conservative).
    """
    caps = caps or {}
    like_cap = float(caps.get("likes", 3.0))
    rt_cap   = float(caps.get("retweets", 3.0))
    rep_cap  = float(caps.get("replies", 2.0))
    
    f_like = 1.0
    f_rt   = 1.0
    f_rep  = 1.0

    # Strong combo: trending + CTA/question → big upside
    if cues.get("trending", 0) == 1 and cues.get("qa_cta", 0) == 1:
        f_like += 0.6
        f_rt   += 0.6
        f_rep  += 0.4

    # Short, punchy, positive aphorism → tends to pop
    if cues.get("short_positive", 0) == 1:
        f_like += 0.4
        f_rt   += 0.4

    # News link → can amplify reach
    if cues.get("has_news_url", 0) == 1:
        f_like += 0.3
        f_rt   += 0.3

    # Big audience + trending/short_positve → extra headroom
    if followers >= 100_000 and (cues.get("trending", 0) == 1 or cues.get("short_positive", 0) == 1):
        f_like += 0.2
        f_rt   += 0.2

    # Penalize external non-news link a bit
    if cues.get("has_url", 0) == 1 and cues.get("has_news_url", 0) == 0:
        f_like = max(1.0, f_like - 0.3)
        f_rt   = max(1.0, f_rt - 0.3)

    # Replies: boost mainly for questions/CTAs
    if cues.get("qa_cta", 0) == 1:
        f_rep += 0.5

    # Caps
    f_like = min(like_cap, f_like)
    f_rt   = min(rt_cap,   f_rt)
    f_rep  = min(rep_cap,  f_rep)

    return {"likes": f_like, "retweets": f_rt, "replies": f_rep}


def _apply_viral_ceiling(mid_likes: int, mid_rt: int, mid_reply: int,
                         curr_high_likes: int, curr_high_rt: int, curr_high_reply: int,
                         cues: dict, followers: int, caps: dict = None) -> tuple[int, int, int]:
    """
    Only raise uppers, never lower them. Uses mid * viral_factor as a ceiling candidate.
    """
    f = _viral_ceiling_factors(cues, followers, caps)
    v_like = int(round(mid_likes  * f["likes"]))
    v_rt   = int(round(mid_rt     * f["retweets"]))
    v_rep  = int(round(mid_reply  * f["replies"]))
    return max(curr_high_likes, v_like), max(curr_high_rt, v_rt), max(curr_high_reply, v_rep)


# --- Timing CSV (one row per /predict) ---
TIMING_CSV_PATH = os.getenv("TWEEXTER_TIMING_CSV", "logs/predict_timing.csv")  # NEW

# --- Simple timing log setup ---
LOG_FILE = "predict_timing.csv"
LOG_HEADER = [
    "ts", "duration_ms", "endpoint", "followers", "text_len",
    "likes_mid", "retweets_mid", "replies_mid", "weights_mode"
]

def _ensure_timing_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(LOG_HEADER)

def _append_timing_row(duration_ms: float, endpoint: str, followers: int, text_len: int,
                       likes_mid: int, retweets_mid: int, replies_mid: int, weights_mode):
    _ensure_timing_log()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.utcnow().isoformat(timespec="seconds"),
            int(round(duration_ms)),
            endpoint,
            int(followers),
            int(text_len),
            int(likes_mid),
            int(retweets_mid),
            int(replies_mid),
            weights_mode or ""
        ])

def _append_timing_csv(row: dict) -> None:
    """
    Append a single row to the timing CSV. Creates folder and header automatically.
    """
    try:
        os.makedirs(os.path.dirname(TIMING_CSV_PATH), exist_ok=True)
    except Exception:
        # if path is just a filename with no directory
        pass

    file_exists = os.path.exists(TIMING_CSV_PATH)
    fieldnames = [
        "ts", "followers", "text_len",
        "likes_mid", "rts_mid", "replies_mid",
        "likes_low", "likes_high", "rts_low", "rts_high", "replies_low", "replies_high",
        "feat_ms", "ml_ms", "persona_ms", "calibration_ms",
        "scaling_ms", "ranges_ms",
        "predict_blended_total_ms", "route_total_ms",
        "models_likes", "models_retweets", "models_replies",
        "weights_mode"
    ]

    # open with newline='' per csv docs to avoid extra blank lines on Windows
    with open(TIMING_CSV_PATH, mode="a", encoding="utf-8", newline="") as f:  # NEW
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        # only write columns we know; ignore extras
        safe_row = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(safe_row)


# ---- shared compute so both JSON + FORM routes can call the same code ----
def _compute(text: str, followers: int, return_details: bool) -> PredictResponse:
    t_route0 = time.perf_counter()  # NEW

    override_w = pick_blend_weights(followers)
    t_pred0 = time.perf_counter()  # NEW
    base = predict_blended(text, override_w)
    t_pred_ms = int((time.perf_counter() - t_pred0) * 1000)  # total predict_blended wall time (backup)
    blended = base.get("blended", {"likes": 0.0, "retweets": 0.0, "replies": 0.0})

    # Apply follower scaling first to get baseline predictions
    cfg = load_cfg()
    
    t_scale0 = time.perf_counter()  # NEW
    adjusted = apply_follower_scaling(blended, followers, cfg)
    scaling_ms = int((time.perf_counter() - t_scale0) * 1000)  # NEW
    
    out_likes    = int(round(adjusted.get("likes", 0.0)))
    out_retweets = int(round(adjusted.get("retweets", 0.0)))
    out_replies  = int(round(adjusted.get("replies", 0.0)))

    ranges_cfg = cfg.get("ranges", {})
    ranges_enabled = bool(ranges_cfg.get("enabled", True))
    range_source = str(ranges_cfg.get("source", "backend")).lower()
    viral_upper_enabled = bool(ranges_cfg.get("viral_upper_enabled", True))
    
    # Build ranges and time them
    t_ranges0 = time.perf_counter()  # NEW
    if (not ranges_enabled) or (range_source != "backend"):
        # Return mid-only; frontend can compute ranges using same rules
        ranges = {
            "likes":    {"low": out_likes,    "mid": out_likes,    "high": out_likes},
            "retweets": {"low": out_retweets, "mid": out_retweets, "high": out_retweets},
            "replies":  {"low": out_replies,  "mid": out_replies,  "high": out_replies},
        }
        ranges_ms = int((time.perf_counter() - t_ranges0) * 1000)  # NEW
        route_total_ms = int((time.perf_counter() - t_route0) * 1000)  # NEW
        
        # --- Append timing CSV (NEW) ---
        btim = base.get("timings", {})  # returned from final_prediction
        _append_timing_csv({
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "followers": followers,
            "text_len": len(text or ""),
            "likes_mid": out_likes,
            "rts_mid": out_retweets,
            "replies_mid": out_replies,
            "likes_low": ranges["likes"]["low"],
            "likes_high": ranges["likes"]["high"],
            "rts_low": ranges["retweets"]["low"],
            "rts_high": ranges["retweets"]["high"],
            "replies_low": ranges["replies"]["low"],
            "replies_high": ranges["replies"]["high"],
            "feat_ms": btim.get("feature_extract_ms", ""),
            "ml_ms": btim.get("ml_predict_ms", ""),
            "persona_ms": btim.get("persona_sim_ms", ""),
            "calibration_ms": btim.get("blend_calibration_ms", ""),
            "scaling_ms": scaling_ms,
            "ranges_ms": ranges_ms,
            "predict_blended_total_ms": btim.get("total_predict_blended_ms", t_pred_ms),
            "route_total_ms": route_total_ms,
            "models_likes": base.get("models_used", {}).get("likes", ""),
            "models_retweets": base.get("models_used", {}).get("retweets", ""),
            "models_replies": base.get("models_used", {}).get("replies", ""),
            "weights_mode": base.get("weights_mode", "")
        })
        
        return PredictResponse(
            likes=out_likes, retweets=out_retweets, replies=out_replies,
            ranges=ranges,
            details=None if not return_details else {
                "note": "Ranges disabled or delegated to frontend",
                "config_snapshot": cfg
            }
        )

    # ====== RANGES: Steps 3–5 ======
    # Floors & scales from config (fall back to current constants if missing)
    floors_cfg = ranges_cfg.get("floors", {})
    LIKE_BAND_FLOOR_CFG = int(floors_cfg.get("likes", LIKE_BAND_FLOOR))
    RT_BAND_MIN_ABS_CFG = int(floors_cfg.get("retweets", RT_BAND_MIN_ABS))
    REPLY_BAND_MIN_ABS_CFG = int(floors_cfg.get("replies", REPLY_BAND_MIN_ABS))

    rt_scale_cfg    = float(ranges_cfg.get("retweet_band_scale", RETWEET_BAND_SCALE))
    reply_scale_cfg = float(ranges_cfg.get("reply_band_scale",   REPLY_BAND_SCALE))

    # Step 2 band already chosen by followers; now customize per metric (step 3)
    base_rel_band = range_band_for_followers(followers)

    # Metric-specific relative bands + absolute floors
    likes_rel_band = base_rel_band
    rt_rel_band    = base_rel_band * rt_scale_cfg
    reply_rel_band = base_rel_band * reply_scale_cfg

    # Step 4: Blend with model uncertainty (if p10/p90 available), take wider band
    floors = {"likes": LIKE_BAND_FLOOR_CFG, "retweets": RT_BAND_MIN_ABS_CFG, "replies": REPLY_BAND_MIN_ABS_CFG}
    likes_rel_band = _blend_with_uncertainty("likes",    out_likes,    likes_rel_band, floors, base)
    rt_rel_band    = _blend_with_uncertainty("retweets", out_retweets, rt_rel_band,    floors, base)
    reply_rel_band = _blend_with_uncertainty("replies",  out_replies,  reply_rel_band, floors, base)

    # Build preliminary ranges from these bands
    likes_low, likes_high = _make_range_from_band(out_likes,    likes_rel_band, LIKE_BAND_FLOOR_CFG)
    rt_low,    rt_high    = _make_range_from_band(out_retweets, rt_rel_band,    RT_BAND_MIN_ABS_CFG)
    reply_low, reply_high = _make_range_from_band(out_replies,  reply_rel_band, REPLY_BAND_MIN_ABS_CFG)

    # Step 5: Content cues shape ONLY the uppers (upside and downside)
    cues = _content_cues(text)
    cue_bumps = ranges_cfg.get("cue_bumps", {})
    likes_high, rt_high, reply_high = _apply_content_cues_to_highs(
        likes_high, rt_high, reply_high,
        out_likes,  out_retweets, out_replies,
        cues, cue_bumps
    )

    # Step 6: Viral ceiling (selective, uppers only)
    if viral_upper_enabled:
        viral_caps = ranges_cfg.get("viral_caps", {})
        likes_high, rt_high, reply_high = _apply_viral_ceiling(
            out_likes, out_retweets, out_replies,
            likes_high, rt_high, reply_high,
            cues, followers, viral_caps
        )

    # Step 7: Cross-metric consistency (RE-APPLY caps after viral)
    rt_cap_ratio = float(cfg.get("retweet_like_cap", 0.95))
    # Replies cap adapts to followers; bump a bit if clear CTA/question
    reply_ratio_cap = reply_like_ratio_for(followers, cfg)
    if cues.get("qa_cta", 0) == 1:
        reply_ratio_cap = min(0.35, reply_ratio_cap * 1.25)

    rt_high    = min(rt_high,    int(round(likes_high * rt_cap_ratio)))
    reply_high = min(reply_high, int(round(likes_high * reply_ratio_cap)))

    # Final clamp: ensure low ≤ mid ≤ high for each metric
    likes_low  = min(likes_low,  out_likes);   likes_high  = max(likes_high,  out_likes)
    rt_low     = min(rt_low,     out_retweets); rt_high    = max(rt_high,     out_retweets)
    reply_low  = min(reply_low,  out_replies);  reply_high = max(reply_high,  out_replies)

    # Package ranges for response
    ranges = {
        "likes":    {"low": likes_low,  "mid": out_likes,    "high": likes_high},
        "retweets": {"low": rt_low,     "mid": out_retweets, "high": rt_high},
        "replies":  {"low": reply_low,  "mid": out_replies,  "high": reply_high},
    }

    ranges_ms = int((time.perf_counter() - t_ranges0) * 1000)  # NEW
    route_total_ms = int((time.perf_counter() - t_route0) * 1000)  # NEW

    # --- Append timing CSV (NEW) ---
    btim = base.get("timings", {})  # returned from final_prediction
    _append_timing_csv({
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "followers": followers,
        "text_len": len(text or ""),
        "likes_mid": out_likes,
        "rts_mid": out_retweets,
        "replies_mid": out_replies,
        "likes_low": ranges["likes"]["low"],
        "likes_high": ranges["likes"]["high"],
        "rts_low": ranges["retweets"]["low"],
        "rts_high": ranges["retweets"]["high"],
        "replies_low": ranges["replies"]["low"],
        "replies_high": ranges["replies"]["high"],
        "feat_ms": btim.get("feature_extract_ms", ""),
        "ml_ms": btim.get("ml_predict_ms", ""),
        "persona_ms": btim.get("persona_sim_ms", ""),
        "calibration_ms": btim.get("blend_calibration_ms", ""),
        "scaling_ms": scaling_ms,
        "ranges_ms": ranges_ms,
        "predict_blended_total_ms": btim.get("total_predict_blended_ms", t_pred_ms),
        "route_total_ms": route_total_ms,
        "models_likes": base.get("models_used", {}).get("likes", ""),
        "models_retweets": base.get("models_used", {}).get("retweets", ""),
        "models_replies": base.get("models_used", {}).get("replies", ""),
        "weights_mode": base.get("weights_mode", "")
    })

    payload = None
    if return_details:
        scales = {m: factor_for(m, followers, cfg) for m in ("likes", "retweets", "replies")}
        payload = {
            "followers": followers,
            "weights_used": base.get("weights_used", {}),
            "models_used": base.get("models_used", {}),
            "ml_raw": base.get("ml", {}),
            "persona_raw": base.get("persona", {}),
            "blended_before_scaling": blended,
            "scaling_factors": scales,
            "follower_baselines": baselines_for(followers, cfg),
            "baseline_weight": baseline_weight(followers, cfg),
            "config_snapshot": cfg,
        }

        # (optional) enrich details with range debug
        payload.update({
            "range_band_base": base_rel_band,
            "range_bands_used": {
                "likes": likes_rel_band, "retweets": rt_rel_band, "replies": reply_rel_band
            },
            "content_cues": cues,
        })

    return PredictResponse(
        likes=out_likes,
        retweets=out_retweets,
        replies=out_replies,
        ranges=ranges,                   # NEW
        details=payload if return_details else None,
    )


# ---------- Routes ----------

@app.get("/healthz")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/")
def root() -> Dict[str, str]:
    return {"name": "Tweexter API", "version": app.version}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict engagement for a single tweet.
    """
    t0 = time.perf_counter()
    try:
        resp = await run_in_threadpool(lambda: _compute(req.text, req.followers, req.return_details))
        elapsed_ms = int(round((time.perf_counter() - t0) * 1000))

        # pull a few details for the log (guarded in case details are off)
        details = getattr(resp, "details", None) or {}
        weights_mode = details.get("weights_used") and details.get("weights_mode", "")
        models_used = details.get("models_used", {}) if isinstance(details.get("models_used", {}), dict) else {}
        _log_timing_row({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_ms": elapsed_ms,
            "followers": req.followers,
            "text_len": len(req.text or ""),
            "likes_mid": resp.likes,
            "retweets_mid": resp.retweets,
            "replies_mid": resp.replies,
            "weights_mode": details.get("weights_mode", ""),
            "models_likes": models_used.get("likes", ""),
            "models_retweets": models_used.get("retweets", ""),
            "models_replies": models_used.get("replies", ""),
        })
        return resp
    except Exception as e:
        # still log a timing row on failure (with mids blank)
        elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
        _log_timing_row({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_ms": elapsed_ms,
            "followers": req.followers,
            "text_len": len(req.text or ""),
            "likes_mid": "",
            "retweets_mid": "",
            "replies_mid": "",
            "weights_mode": "error",
            "models_likes": "",
            "models_retweets": "",
            "models_replies": "",
        })
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e


@app.post("/predict-form", response_model=PredictResponse)
async def predict_form(
    text: str = Form(..., description="Paste tweet text (multiline OK)"),
    followers: int = Form(..., ge=1, description="Author follower count"),
    return_details: bool = Form(False, description="Include full breakdown"),
) -> PredictResponse:
    t0 = time.perf_counter()
    try:
        resp = await run_in_threadpool(lambda: _compute(text, followers, return_details))
        elapsed_ms = int(round((time.perf_counter() - t0) * 1000))

        # pull a few details for the log (guarded in case details are off)
        details = getattr(resp, "details", None) or {}
        weights_mode = details.get("weights_used") and details.get("weights_mode", "")
        models_used = details.get("models_used", {}) if isinstance(details.get("models_used", {}), dict) else {}
        _log_timing_row({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_ms": elapsed_ms,
            "followers": followers,
            "text_len": len(text or ""),
            "likes_mid": resp.likes,
            "retweets_mid": resp.retweets,
            "replies_mid": resp.replies,
            "weights_mode": details.get("weights_mode", ""),
            "models_likes": models_used.get("likes", ""),
            "models_retweets": models_used.get("retweets", ""),
            "models_replies": models_used.get("replies", ""),
        })
        return resp
    except Exception as e:
        # still log a timing row on failure (with mids blank)
        elapsed_ms = int(round((time.perf_counter() - t0) * 1000))
        _log_timing_row({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_ms": elapsed_ms,
            "followers": followers,
            "text_len": len(text or ""),
            "likes_mid": "",
            "retweets_mid": "",
            "replies_mid": "",
            "weights_mode": "error",
            "models_likes": "",
            "models_retweets": "",
            "models_replies": "",
        })
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e
