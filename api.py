# api.py
# FastAPI wrapper for Tweexter predictions with follower-aware scaling.

import re  # NEW
import warnings
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel, Field, validator
from starlette.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

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
    followers: int = Field(..., ge=1, description="Author follower count")
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


# ---- shared compute so both JSON + FORM routes can call the same code ----
def _compute(text: str, followers: int, return_details: bool) -> PredictResponse:
    override_w = pick_blend_weights(followers)
    base = predict_blended(text, override_w)
    blended = base.get("blended", {"likes": 0.0, "retweets": 0.0, "replies": 0.0})

    # Apply follower scaling first to get baseline predictions
    cfg = load_cfg()
    adjusted = apply_follower_scaling(blended, followers, cfg)
    
    out_likes    = int(round(adjusted.get("likes", 0.0)))
    out_retweets = int(round(adjusted.get("retweets", 0.0)))
    out_replies  = int(round(adjusted.get("replies", 0.0)))

    ranges_cfg = cfg.get("ranges", {})
    ranges_enabled = bool(ranges_cfg.get("enabled", True))
    range_source = str(ranges_cfg.get("source", "backend")).lower()
    viral_upper_enabled = bool(ranges_cfg.get("viral_upper_enabled", True))
    
    if (not ranges_enabled) or (range_source != "backend"):
        # Return mid-only; frontend can compute ranges using same rules
        ranges = {
            "likes":    {"low": out_likes,    "mid": out_likes,    "high": out_likes},
            "retweets": {"low": out_retweets, "mid": out_retweets, "high": out_retweets},
            "replies":  {"low": out_replies,  "mid": out_replies,  "high": out_replies},
        }
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
    try:
        return await run_in_threadpool(lambda: _compute(req.text, req.followers, req.return_details))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e


@app.post("/predict-form", response_model=PredictResponse)
async def predict_form(
    text: str = Form(..., description="Paste tweet text (multiline OK)"),
    followers: int = Form(..., ge=1, description="Author follower count"),
    return_details: bool = Form(False, description="Include full breakdown"),
) -> PredictResponse:
    try:
        return await run_in_threadpool(lambda: _compute(text, followers, return_details))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e
