# final.py
# Wrapper that asks for follower count and applies sublinear follower scaling
# to Tweexter's blended predictions from final_prediction.py.

import sys, json, math, inspect
from pathlib import Path

from final_prediction import predict_blended  # your existing function

CFG_PATH = Path("follower_scaling.json")

# --- defaults aligned to our research ---
DEFAULT_CFG = {
    "F0": 10000.0,  # reference followers
    "alpha": {      # sublinear exponents (engagement ~ followers^alpha)
        "likes": 0.63,
        "retweets": 0.73,
        "replies": 0.55
    },
    "min_factor": 0.20,  # clamp extreme scaling
    "max_factor": 50.0,
    # sanity constraints to keep metrics realistic relative to each other
    "retweet_like_cap": 0.95,  # retweets should rarely exceed likes
    # replies are a small share of likes that grows slowly with audience size
    "reply_ratio": {"min": 0.01, "max": 0.18, "k": 1.2, "pivot": 50000},

    # --- RANGES / FEATURE FLAGS (used by API) ---
    "ranges": {
        "enabled": True,                 # master switch for ranges
        "source": "backend",             # "backend" | "frontend"
        "viral_upper_enabled": True,     # allow viral ceiling bump

        # follower-tier → default ±band (relative)
        "tier_bands": [
            {"max_followers": 1000,     "band": 0.50},
            {"max_followers": 5000,     "band": 0.35},
            {"max_followers": 10000,    "band": 0.30},
            {"max_followers": 50000,    "band": 0.25},
            {"max_followers": 100000,   "band": 0.22},
            {"max_followers": 300000,   "band": 0.20},
            {"max_followers": 600000,   "band": 0.18},
            {"max_followers": 1000000,  "band": 0.16},
            {"max_followers": None,     "band": 0.15}
        ],

        # (10) Small-number safeguards → absolute floors for bands
        "floors": {"likes": 8, "retweets": 5, "replies": 2},

        # Per-metric band scaling vs likes band
        "retweet_band_scale": 0.70,
        "reply_band_scale":   0.50,

        # Content-cue bumps (multiplicative on UPPER bounds, e.g. +0.15 = +15%)
        "cue_bumps": {
            "trending":        0.15,
            "news":            0.10,
            "short_positive":  0.10,
            "qa_cta_like":     0.05,   # small like bump when Q/CTA
            "qa_cta_reply":    0.25,   # bigger reply bump when Q/CTA
            "nonnews_link":   -0.10    # shave upper if external non-news link
        },

        # Viral caps for ceiling rule (step 6): mid * cap
        "viral_caps": {"likes": 3.0, "retweets": 3.0, "replies": 2.0}
    }
}

# NEW: follower-tier blend weights
def pick_blend_weights(followers: int):
    try:
        cfg = json.loads(Path("blend_weights.json").read_text(encoding="utf-8"))
    except Exception:
        return None

    if cfg.get("force_static", False):
        g = cfg.get("global", {})
        return {
            "likes": float(g.get("likes", 0.90)),
            "retweets": float(g.get("retweets", 1.00)),
            "replies": float(g.get("replies", 0.20)),
        }

    tiers = cfg.get("tiers", [])
    for t in tiers:
        mx = t.get("max_followers")
        if (mx is None) or (followers <= int(mx)):
            return {
                "likes": float(t["likes"]),
                "retweets": float(t["retweets"]),
                "replies": float(t["replies"]),
            }
    return None

def load_cfg():
    cfg = json.loads(json.dumps(DEFAULT_CFG))  # deep copy
    if CFG_PATH.exists():
        try:
            user = json.loads(CFG_PATH.read_text(encoding="utf-8"))
            if "F0" in user: cfg["F0"] = float(user["F0"])
            if "alpha" in user and isinstance(user["alpha"], dict):
                for k in ("likes", "retweets", "replies"):
                    if k in user["alpha"]:
                        cfg["alpha"][k] = float(user["alpha"][k])
            if "min_factor" in user: cfg["min_factor"] = float(user["min_factor"])
            if "max_factor" in user: cfg["max_factor"] = float(user["max_factor"])
            if "retweet_like_cap" in user: cfg["retweet_like_cap"] = float(user["retweet_like_cap"])
            if "reply_ratio" in user and isinstance(user["reply_ratio"], dict):
                for k in ("min","max","k","pivot"):
                    if k in user["reply_ratio"]:
                        cfg["reply_ratio"][k] = float(user["reply_ratio"][k])
            
            # inside load_cfg(), after existing keys are merged from user json:
            for key in ("visibility_rate", "like_rate_per_view", "rt_like_ratio", "reply_like_ratio"):
                if key in user and isinstance(user[key], dict):
                    cfg[key] = {}  # ensure the key exists
                    for k in ("min", "max", "k", "pivot"):
                        if k in user[key]:
                            cfg[key][k] = float(user[key][k])

            if "baseline_blend" in user and isinstance(user["baseline_blend"], dict):
                cfg["baseline_blend"] = {}  # ensure the key exists
                for k in ("w_small", "w_big", "pivot"):
                    if k in user["baseline_blend"]:
                        cfg["baseline_blend"][k] = float(user["baseline_blend"][k])

            # --- RANGES / FEATURE FLAGS (deep-merge) ---
            if "ranges" in user and isinstance(user["ranges"], dict):
                # start from defaults
                base_ranges = cfg.get("ranges", {}).copy()
                for k, v in user["ranges"].items():
                    if isinstance(v, dict):
                        base_ranges[k] = {**base_ranges.get(k, {}), **v}
                    else:
                        base_ranges[k] = v
                cfg["ranges"] = base_ranges
        except Exception as e:
            print(f"⚠️ follower_scaling.json load failed; using defaults. Error: {e}")
    return cfg

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

# --- follower-aware baselines on log10(N) ---
def _logistic_on_logN(n: float, lo: float, hi: float, k: float, pivot: float) -> float:
    x = math.log10(max(1.0, n)) - math.log10(max(1.0, pivot))
    return lo + (hi - lo) * _sigmoid(k * x)

def visibility_rate_for(followers: float, cfg: dict) -> float:
    vr = cfg["visibility_rate"]
    return _logistic_on_logN(followers, vr["min"], vr["max"], vr["k"], vr["pivot"])

def like_rate_per_view_for(followers: float, cfg: dict) -> float:
    lr = cfg["like_rate_per_view"]
    return _logistic_on_logN(followers, lr["min"], lr["max"], lr["k"], lr["pivot"])

def rt_like_ratio_for(followers: float, cfg: dict) -> float:
    rr = cfg["rt_like_ratio"]
    return _logistic_on_logN(followers, rr["min"], rr["max"], rr["k"], rr["pivot"])

def reply_like_ratio_for(followers: float, cfg: dict) -> float:
    rr = cfg["reply_like_ratio"]
    return _logistic_on_logN(followers, rr["min"], rr["max"], rr["k"], rr["pivot"])

def baselines_for(followers: float, cfg: dict) -> dict:
    vr  = visibility_rate_for(followers, cfg)
    lrv = like_rate_per_view_for(followers, cfg)
    base_likes = max(1.0, followers * vr * lrv)
    base_retweets = base_likes * rt_like_ratio_for(followers, cfg)
    base_replies  = base_likes * reply_like_ratio_for(followers, cfg)
    return {"likes": base_likes, "retweets": base_retweets, "replies": base_replies}

def baseline_weight(followers: float, cfg: dict) -> float:
    b = cfg["baseline_blend"]
    # more weight on baseline for larger audiences
    x = math.log10(max(1.0, followers)) - math.log10(max(1.0, b["pivot"]))
    return float(b["w_small"] + (b["w_big"] - b["w_small"]) * _sigmoid(1.2 * x))

def factor_for(metric: str, followers: float, cfg: dict) -> float:
    # power-law scaling: (N/F0)^alpha, with clamps
    n = max(1.0, float(followers))
    F0 = max(1.0, float(cfg["F0"]))
    alpha = float(cfg["alpha"][metric])
    f = (n / F0) ** alpha
    return max(cfg["min_factor"], min(cfg["max_factor"], f))

def reply_ratio_for(followers: float, cfg: dict) -> float:
    # replies/likes ratio rises slowly with audience size; logistic on log10(N)
    rr = cfg["reply_ratio"]
    lo, hi, k, pivot = rr["min"], rr["max"], rr["k"], max(1.0, rr["pivot"])
    x = math.log10(max(1.0, followers)) - math.log10(pivot)
    return lo + (hi - lo) * _sigmoid(k * x)

def apply_follower_scaling(blended: dict, followers: float, cfg: dict) -> dict:
    # 1) power-law scaling per metric (dampened by cfg alpha/F0)
    scaled = {}
    for m in ("likes", "retweets", "replies"):
        base = float(blended.get(m, 0.0))
        scaled[m] = base * factor_for(m, followers, cfg)

    # 2) follower-aware baselines
    base = baselines_for(followers, cfg)
    w = baseline_weight(followers, cfg)  # 0.25..0.55

    # 3) geometric shrink toward baseline
    blended_out = {}
    for m in ("likes", "retweets", "replies"):
        s = max(0.0, scaled[m])
        b = max(1.0, base[m])
        blended_out[m] = (s ** (1.0 - w)) * (b ** w)

    # 4) cross-metric sanity
    likes = blended_out["likes"]
    # retweets should not exceed cap relative to likes
    blended_out["retweets"] = min(blended_out["retweets"], likes * float(cfg["retweet_like_cap"]))
    # replies anchored by follower-aware like ratio (soft cap)
    replies_cap = likes * reply_like_ratio_for(followers, cfg)
    blended_out["replies"] = min(blended_out["replies"], replies_cap)

    return blended_out

def main():
    if len(sys.argv) < 2:
        print('Usage: python final.py "your tweet text here"')
        sys.exit(1)

    tweet = sys.argv[1]

    # ask follower count once in CLI
    try:
        fc_raw = input("Enter follower count (integer, required for scaling): ").strip()
    except EOFError:
        fc_raw = ""
    if not fc_raw:
        print("⚠️ No follower count provided. Please run again and enter a number.")
        sys.exit(2)

    try:
        followers = int(fc_raw.replace(",", "").strip())
    except Exception:
        print("⚠️ Invalid follower count. Please provide an integer like 15000.")
        sys.exit(3)

    # call your base predictor
    override_w = pick_blend_weights(followers)
    res = predict_blended(tweet, override_w)
    blended = res.get("blended", {})

    # load scaling config and apply
    cfg = load_cfg()
    adjusted = apply_follower_scaling(blended, followers, cfg)

    # show both for transparency
    print(f"\nFollower count used: {followers:,} | "
          f"F0={int(cfg['F0'])} | α={cfg['alpha']} | "
          f"reply_ratio~[{cfg['reply_ratio']['min']:.02f}-{cfg['reply_ratio']['max']:.02f}]")

    print("\n=== BLENDED PREDICTIONS (original) ===")
    for k in ("likes","retweets","replies"):
        v = blended.get(k, 0.0)
        print(f"{k.capitalize():9s}: {int(round(v))}")

    print("\n=== BLENDED PREDICTIONS (followers-adjusted) ===")
    for k in ("likes","retweets","replies"):
        v = adjusted.get(k, 0.0)
        print(f"{k.capitalize():9s}: {int(round(v))}")

if __name__ == "__main__":
    main()
