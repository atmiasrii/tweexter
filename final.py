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
    "reply_ratio": {"min": 0.01, "max": 0.18, "k": 1.2, "pivot": 50000}
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
        except Exception as e:
            print(f"⚠️ follower_scaling.json load failed; using defaults. Error: {e}")
    return cfg

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

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
    # 1) raw power-law scaling per metric
    scaled = {}
    for m in ("likes", "retweets", "replies"):
        base = float(blended.get(m, 0.0))
        scaled[m] = base * factor_for(m, followers, cfg)

    # 2) cross-metric consistency constraints
    likes = max(0.0, scaled.get("likes", 0.0))
    # retweets should not exceed a high fraction of likes
    rt_cap = likes * float(cfg["retweet_like_cap"])
    scaled["retweets"] = max(0.0, min(scaled.get("retweets", 0.0), rt_cap))

    # replies proportion target vs likes using follower-aware ratio
    rr = reply_ratio_for(followers, cfg)
    replies_cap = likes * rr
    # allow replies up to the cap; keep if model predicted less
    scaled["replies"] = max(0.0, min(scaled.get("replies", 0.0), replies_cap))

    return scaled

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
