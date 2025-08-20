from __future__ import annotations
CHAR_LIMIT = 280
# === Minimal Post-Processing (keep-only) ===
class MiniPost:
    @staticmethod
    def normalize_spaces(s: str) -> str:
        s = re.sub(r'[ \t]+', ' ', s)
        s = re.sub(r'\n{3,}', '\n\n', s)
        return s.strip()

    @staticmethod
    def sanitize_basic(text: str, ent: "Entities") -> str:
        """Reinsert any dropped original entities without inventing new ones."""
        out = (text or "").strip()
        # ensure original mentions at the front if missing
        for m in getattr(ent, "mentions", []) or []:
            if m and m not in out:
                out = f"{m} {out}"
        # ensure original links present (append once)
        for link in getattr(ent, "links", []) or []:
            if link and link not in out:
                out = f"{out} {link}"
        # ensure original hashtags present (append once)
        for h in getattr(ent, "hashtags", []) or []:
            if h and h not in out and len(out) + len(h) + 1 <= CHAR_LIMIT:
                out = f"{out} {h}"
        return out

    @staticmethod
    def clamp_hashtags(text: str, ent: "Entities", max_new: int) -> str:
        """Keep all original hashtags; allow at most `max_new` new ones (drop the rest)."""
        orig = {h.lower() for h in getattr(ent, "hashtags", []) or []}
        toks = text.split()
        kept, new_seen = [], set()
        allowed_new = max(0, int(max_new))
        for t in toks:
            if t.startswith("#"):
                low = t.lower()
                if low in orig:
                    kept.append(t)
                else:
                    if len(new_seen) < allowed_new and low not in new_seen:
                        kept.append(t); new_seen.add(low)
                    # else drop extra new tag
            else:
                kept.append(t)
        return " ".join(kept).strip()

    @staticmethod
    def limit_emojis(text: str, max_emojis: int) -> str:
        if count_emojis(text) <= max_emojis:
            return text
        keep = max_emojis
        def _strip_extra(m):
            nonlocal keep
            if keep > 0:
                keep -= 1
                return m.group(0)
            return ""
        return EMOJI_RE.sub(_strip_extra, text)

    @staticmethod
    def ensure_char_limit(text: str, limit: int = CHAR_LIMIT) -> str:
        return _smart_truncate(text, limit=limit)

    @staticmethod
    def dedupe(texts: List[str]) -> List[str]:
        seen, out = set(), []
        for t in texts:
            key = re.sub(r'\W+', '', (t or "").lower())
            if key in seen: 
                continue
            seen.add(key); out.append(t)
        return out

    @staticmethod
    def batch(
        *,
        original: str,
        candidates: List[str],
        entities: "Entities",
        max_new_hashtags: int,
        max_emojis: int,
        limit: int
    ) -> List[str]:
        """Minimal, deterministic clean-up with only hard constraints."""
        out = []
        for c in candidates:
            if not c or not c.strip():
                continue
            s = _strip_quote_bullets(c)
            s = MiniPost.sanitize_basic(s, entities)

            # Reject if it introduces new links/mentions/numeric claims/years
            if not _no_new_claim_risk(original, s):
                continue

            s = MiniPost.clamp_hashtags(s, entities, max_new_hashtags)
            s = MiniPost.limit_emojis(s, max_emojis)
            s = MiniPost.ensure_char_limit(s, limit)
            s = MiniPost.normalize_spaces(s)
            out.append(s)

        out = MiniPost.dedupe(out)

        # Always return at least one candidate
        if not out:
            fallback = MiniPost.sanitize_basic(original, entities)
            fallback = MiniPost.clamp_hashtags(fallback, entities, max_new_hashtags)
            fallback = MiniPost.limit_emojis(fallback, max_emojis)
            fallback = MiniPost.ensure_char_limit(fallback, limit)
            out = [MiniPost.normalize_spaces(fallback)]

        return out
# improve.py — skeleton (v0)
# Purpose: take a tweet, generate 5–7 improved variants with Gemini, score them
# with your prediction stack/personas, and return the best + context.
#
# Environment Variable Examples (override config.json):
# export SCORING_MODE=blend
# export NUM_VARIANTS=7
# export PERSONA_SAMPLE_SIZE=1200
# export GEMINI_MODEL="models/gemini-2.5-flash"
# export TEMPERATURE=0.9
# export MAX_NEW_HASHTAGS=1
# export MAX_EMOJIS=2
# export ALLOW_CTA=true

 # --- Consolidated imports ---
import re, unicodedata
import os
import json
import time
import random
import logging
import dataclasses
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from functools import lru_cache
from datetime import datetime, timezone
import threading
# --- ADD: scorer warmup utility ---
def wait_for_scorer_ready(timeout=15, interval=0.5, verbose=True):
    """
    Ping the scorer API until it responds, or until timeout (in seconds).
    Returns True if ready, False if not.
    """
    if not SCORER_URL:
        return False
    import time, requests, urllib.parse
    base = SCORER_URL
    # strip /predict if present
    if base.endswith("/predict"):
        base = base.rsplit("/predict", 1)[0]
    url = urllib.parse.urljoin(base + "/", "healthz")
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=1.5)
            if r.status_code == 200:
                if verbose: print(f"[improve.py] Scorer API ready at {url}")
                return True
        except Exception as e:
            last_err = e
            if verbose: print(f"[improve.py] Waiting for scorer API at {url}... ({e})")
        time.sleep(interval)
    if verbose: print(f"[improve.py] Scorer API not ready after {timeout}s: {last_err}")
    return False

# optional HTTP scorer settings (useful when running a separate scorer service)

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



# Define retries for HTTPAdapter
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
)
# Increase HTTPAdapter pool size to avoid connection pool full errors
adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
_SESSION = requests.Session()
_SESSION.mount("http://", adapter)
_SESSION.mount("https://", adapter)

SCORER_URL = os.getenv("SCORER_URL")  # e.g., http://localhost:8000/predict
SCORER_TIMEOUT = float(os.getenv("SCORER_TIMEOUT", "8"))

# Optional ELO endpoint (only if you have it)
SCORER_ELO_URL = os.getenv("SCORER_ELO_URL")  # e.g., http://localhost:8000/elo

# --- ADD: config loader with precedence (ENV > JSON > defaults) ---

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return v.lower() in {"1","true","yes","y","on"}

def _env_float(name: str, default: float) -> float:
    try: return float(os.getenv(name, default))
    except: return default

def _env_int(name: str, default: int) -> int:
    try: return int(os.getenv(name, default))
    except: return default

@dataclass(frozen=True)
class Settings:
    NUM_VARIANTS: int = 30                     # Default is now 30
    SCORING_MODE: str = "blend"                # "ml" | "elo" | "blend"
    ALPHA_RETWEETS: float = 2.0
    BETA_REPLIES: float = 1.0
    MAX_NEW_HASHTAGS: int = 1
    MAX_EMOJIS: int = 2
    ALLOW_CTA: bool = True
    PERSONA_SAMPLE_SIZE: int = 1000            # if sub-sampling for Elo
    TRENDING_HASHTAGS_ENABLED: bool = True
    GEMINI_MODEL: str = "models/gemini-2.5-flash"  # Gemini 2.5 Flash
    TEMPERATURE: float = 0.85
    TIMEOUT: int = 20                          # seconds
    RETRIES: int = 2
    DISABLE_ENGAGEMENT_MECHANICS: bool = True

def load_settings(path: str = "config.json") -> Settings:
    # 1) defaults
    base = Settings().__dict__.copy()

    # 2) overlay file (if present)
    try:
        with open(path, "r") as f:
            data = json.load(f) or {}
        base.update({k: data[k] for k in base.keys() if k in data})
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"⚠️ config.json load error: {e} (using defaults/env)")

    # 3) overlay ENV (wins)
    base["NUM_VARIANTS"]            = _env_int("NUM_VARIANTS", base["NUM_VARIANTS"])
    base["SCORING_MODE"]            = os.getenv("SCORING_MODE", base["SCORING_MODE"])
    base["ALPHA_RETWEETS"]          = _env_float("ALPHA_RETWEETS", base["ALPHA_RETWEETS"])
    base["BETA_REPLIES"]            = _env_float("BETA_REPLIES", base["BETA_REPLIES"])
    base["MAX_NEW_HASHTAGS"]        = _env_int("MAX_NEW_HASHTAGS", base["MAX_NEW_HASHTAGS"])
    base["MAX_EMOJIS"]              = _env_int("MAX_EMOJIS", base["MAX_EMOJIS"])
    base["ALLOW_CTA"]               = _env_bool("ALLOW_CTA", base["ALLOW_CTA"])
    base["PERSONA_SAMPLE_SIZE"]     = _env_int("PERSONA_SAMPLE_SIZE", base["PERSONA_SAMPLE_SIZE"])
    base["TRENDING_HASHTAGS_ENABLED"]= _env_bool("TRENDING_HASHTAGS_ENABLED", base["TRENDING_HASHTAGS_ENABLED"])
    base["GEMINI_MODEL"]            = os.getenv("GEMINI_MODEL", base["GEMINI_MODEL"])
    base["TEMPERATURE"]             = _env_float("TEMPERATURE", base["TEMPERATURE"])
    base["TIMEOUT"]                 = _env_int("TIMEOUT", base["TIMEOUT"])
    base["RETRIES"]                 = _env_int("RETRIES", base["RETRIES"])
    base["DISABLE_ENGAGEMENT_MECHANICS"] = _env_bool("DISABLE_ENGAGEMENT_MECHANICS", base.get("DISABLE_ENGAGEMENT_MECHANICS", True))

    return Settings(**base)


# single global settings object
SETTINGS = load_settings()


# --- Make HF token and persistent cache available to huggingface_hub / transformers ---
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    try:
        with open("config.json") as f:
            HF_TOKEN = (json.load(f) or {}).get("HF_TOKEN")
    except FileNotFoundError:
        HF_TOKEN = None
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN  # huggingface_hub reads this env var
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # faster downloads
# Set persistent cache directory early (before any model loading)
HF_CACHE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".hf_cache"))
os.environ["HF_HOME"] = HF_CACHE_DIR

# Optional: create the cache directory if it doesn't exist
import os
if not os.path.exists(HF_CACHE_DIR):
    os.makedirs(HF_CACHE_DIR, exist_ok=True)

# --- Utility: Pre-download all required HuggingFace models (run once if needed) ---
def predownload_hf_models(model_list):
    """Pre-download all HuggingFace models to the persistent cache."""
    from huggingface_hub import snapshot_download
    auth = {"token": HF_TOKEN} if HF_TOKEN else {}
    for repo in model_list:
        print(f"Downloading {repo} ...")
        snapshot_download(repo_id=repo, local_dir_use_symlinks=False, **auth)
    print("All models downloaded and cached in:", HF_CACHE_DIR)

# PRELOAD_MODELS = [
#     "cardiffnlp/twitter-roberta-base-sentiment-latest",
#     # Add any others you use
# ]
# predownload_hf_models(PRELOAD_MODELS)


# --- WIRE SETTINGS INTO EXISTING KNOBS ---
NUM_VARIANTS_DEFAULT   = SETTINGS.NUM_VARIANTS  # Will be 30 by default
# FORCE BLEND MODE ALWAYS
SCORING_MODE_DEFAULT   = "blend"  # Always use blend mode, ignore config/env
BLEND_W_ML  = 0.70
BLEND_W_ELO = 0.30
ALPHA_RETWEETS         = SETTINGS.ALPHA_RETWEETS
BETA_REPLIES           = SETTINGS.BETA_REPLIES
MAX_NEW_HASHTAGS       = SETTINGS.MAX_NEW_HASHTAGS
MAX_EMOJIS_GENERAL     = SETTINGS.MAX_EMOJIS
ALLOW_CTA_DEFAULT      = SETTINGS.ALLOW_CTA
PERSONA_SAMPLE_SIZE    = SETTINGS.PERSONA_SAMPLE_SIZE
TRENDING_ENABLED       = SETTINGS.TRENDING_HASHTAGS_ENABLED
GEMINI_MODEL           = SETTINGS.GEMINI_MODEL
TEMPERATURE            = SETTINGS.TEMPERATURE
GEN_TIMEOUT_S          = SETTINGS.TIMEOUT
GEN_RETRIES            = SETTINGS.RETRIES

# --- ADD: parallelism knobs ---
# PERFORMANCE OPTIMIZATIONS:
# - ProcessPool for CPU-bound ML scoring (8-16 workers)  
# - ThreadPool for I/O bound tasks (12 workers)
# - LRU cache for repeated text features (sentiment, hashtags, length)
# - TTL cache for trending hashtags (15 min)
# - Persona sampling cap (≤1000) for Elo performance
ML_POOL_SIZE = int(os.getenv("ML_POOL_SIZE", "8"))       # 8–16 recommended
THREAD_POOL_SMALL = int(os.getenv("THREAD_POOL_SMALL", "12"))

# Dedicated pools (lazy init for safety on Windows/notebooks)
ML_EXEC = None  # type: Optional[ProcessPoolExecutor]
AUX_EXEC = ThreadPoolExecutor(max_workers=THREAD_POOL_SMALL)  # minor parallel IO/cleanup

def _get_ml_exec():
    global ML_EXEC
    if ML_EXEC is None:
        from concurrent.futures import ProcessPoolExecutor
        ML_EXEC = ProcessPoolExecutor(max_workers=ML_POOL_SIZE)
    return ML_EXEC

# =========================
# 0) CONFIG & CONSTANTS
# =========================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    try:
        with open("config.json") as f:
            GEMINI_API_KEY = json.load(f).get("GEMINI_API_KEY")
    except FileNotFoundError:
        GEMINI_API_KEY = None

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing. Set it in env or config.json.")

CHAR_LIMIT     = 280

# Updated emoji limits by type
MAX_EMOJIS_NEWS = 1

# --- ADD: blend weights & composite weights ---
BLEND_W_ML  = 0.70
BLEND_W_ELO = 0.30
ELO_PERSONA_CAP = PERSONA_SAMPLE_SIZE  # keep <=1000 for speed

LOG_LEVEL = os.getenv("IMPROVE_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("improve")

# --- ADD: quick language heuristic + entity fallback + precheck ---------

LANG_NON_ASCII_RE = re.compile(r'[^\x00-\x7F]+')
EMAIL_RE = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
PHONE_RE = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b')

# --- ADD: Enhanced regex patterns for guardrails ---
LINK_RE    = re.compile(r'https?://[^\s]+', re.IGNORECASE)
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#\w+')
# basic emoji range coverage (BMP + some supplementary planes)
EMOJI_RE   = re.compile(
    r'['
    r'\U0001F300-\U0001F6FF'  # symbols & pictographs, transport & map
    r'\U0001F700-\U0001F77F'  # alchemical
    r'\U0001F780-\U0001F7FF'  # geometric extended
    r'\U0001F800-\U0001F8FF'  # arrows supplemental
    r'\U0001F900-\U0001F9FF'  # supplemental symbols & pictographs
    r'\U0001FA00-\U0001FAFF'  # extended-A
    r'\u2600-\u27BF'          # misc symbols
    r'\u23F0\u23F3\u231A\u231B'  # common singletons
    r']'
)

# --- Step B detectors ---
HOOK_PREFIXES = (
    "Unpopular opinion:", "Reminder:", "Hot take:", "PSA:", "Heads up:",
    "Fact:", "Myth:", "Reality:", "New:", "Data:", "Update:"
)
QUESTION_GENERIC = {
    "thoughts?", "what do you think?", "agree?", "opinions?",
    "what are your thoughts?", "your thoughts?"
}
BENEFIT_PREFIXES = ("Payoff:", "The win:", "You get:", "Outcome:", "What you’ll get:")
CURIOSITY_TOKENS = ("Ever wonder", "Here’s the twist:", "The catch:", "Counterintuitive")
CONTRAST_TOKENS  = (" vs ", "Before", "After", "Old way", "New way", "→", "vs.")

def _starts_with_number(s: str) -> bool:
    return bool(re.match(r'^\s*[\d#@]', s))

def _has_prefix(s: str, prefixes) -> bool:
    t = s.strip()
    return any(t.startswith(p) for p in prefixes)

def is_hook_variant(s: str) -> bool:
    return _starts_with_number(s) or _has_prefix(s, HOOK_PREFIXES)

def is_specific_question(s: str, original: str) -> bool:
    if "?" not in s: return False
    t = s.lower()
    if any(p in t for p in QUESTION_GENERIC): return False
    ent = Entities.from_text(original)
    return bool(ent.hashtags or ent.mentions or KEY_NUMBER_RE.search(original))

def is_benefit_first(s: str) -> bool:
    return _has_prefix(s, BENEFIT_PREFIXES)

def has_curiosity_gap(s: str) -> bool:
    return any(tok in s for tok in CURIOSITY_TOKENS)

def has_contrast(s: str) -> bool:
    return any(tok in s for tok in CONTRAST_TOKENS)

# Numbers/dates (enhanced from earlier)
KEY_NUMBER_RE = re.compile(r'(?<![\w.])(?:\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+%)(?![\w.])')
YEAR_RE       = re.compile(r'(?:19|20)\d{2}')

def _detect_language_heuristic(text: str) -> str:
    return "non_en" if LANG_NON_ASCII_RE.search(text) else "en"

def unicode_len(s: str) -> int:
    # Python 3 len() counts code points; good enough for 280-char cap.
    return len(s)

def count_emojis(s: str) -> int:
    return len(EMOJI_RE.findall(s))

def extract_entities(text: str) -> Entities:
    links = LINK_RE.findall(text)
    mentions = MENTION_RE.findall(text)
    hashtags = HASHTAG_RE.findall(text)
    return Entities(links, mentions, hashtags)

# very light script heuristic to keep original language/script
SCRIPT_PATTERNS = {
    "latin":    re.compile(r'[A-Za-z]'),
    "cyrillic": re.compile(r'[А-Яа-я]'),
    "greek":    re.compile(r'[Α-Ωα-ω]'),
    "han":      re.compile(r'[\u4E00-\u9FFF]'),
    "kana":     re.compile(r'[\u3040-\u30FF]'),
    "arabic":   re.compile(r'[\u0600-\u06FF]'),
    "devanagari": re.compile(r'[\u0900-\u097F]'),
}

def _dominant_script(text: str) -> str:
    counts = {k: len(p.findall(text)) for k, p in SCRIPT_PATTERNS.items()}
    # fall back to latin if all zero
    return max(counts, key=counts.get) if any(counts.values()) else "latin"

def _same_language(original: str, candidate: str) -> bool:
    # Heuristic: keep dominant script the same
    return _dominant_script(original) == _dominant_script(candidate)

def _no_new_claim_risk(original: str, candidate: str) -> bool:
    """
    Safety delta check:
    - don't add new numbers/dates that weren't in original
    - don't add new @mentions or links
    - keep tone away from medical/financial prescriptive verbs if not present
    """
    o_ent, c_ent = extract_entities(original), extract_entities(candidate)
    # new links/mentions not allowed
    if set(c_ent.links) - set(o_ent.links):
        return False
    if set(c_ent.mentions) - set(o_ent.mentions):
        return False
    # new numeric claims (allow formatting differences but not new tokens)
    o_nums, c_nums = set(KEY_NUMBER_RE.findall(original)), set(KEY_NUMBER_RE.findall(candidate))
    if c_nums - o_nums:
        return False
    o_years, c_years = set(YEAR_RE.findall(original)), set(YEAR_RE.findall(candidate))
    if c_years - o_years:
        return False
    # prescriptive advice guard (very light)
    risk_words = {'diagnose','cure','prescribe','invest','buy','sell','short','leverage'}
    if (risk_words & set(w.lower().strip('.,!?') for w in candidate.split())
        and not (risk_words & set(w.lower().strip('.,!?') for w in original.split()))):
        return False
    return True

def _passes_all_constraints(
    candidate: str,
    original: str,
    entities: Entities,
    tweet_type: str,
    max_new_hashtags: int = MAX_NEW_HASHTAGS
) -> Tuple[bool, List[str]]:
    """Master constraint checker - verifies all guardrails are met."""
    violations = []
    
    # Length and character constraints
    if unicode_len(candidate) > CHAR_LIMIT:
        violations.append("char_limit_exceeded")
    
    # Language/script preservation
    if not _same_language(original, candidate):
        violations.append("language_changed")
    
    # Entity preservation
    orig_entities = entities
    cand_entities = extract_entities(candidate)
    
    # Check links preservation
    for link in orig_entities.links:
        if link and link not in candidate:
            violations.append("missing_link")
            break
    
    # Check mentions preservation
    for mention in orig_entities.mentions:
        if mention and mention not in candidate:
            violations.append("missing_mention")
            break
    
    # Check hashtags preservation
    for hashtag in orig_entities.hashtags:
        if hashtag and hashtag not in candidate:
            violations.append("missing_hashtag")
            break
    
    # Key facts preservation
    key_facts = _extract_key_facts(original)
    for number in key_facts["numbers"]:
        if number and number not in candidate:
            violations.append("missing_number")
            break
    
    for year in key_facts["years"]:
        if year and year not in candidate:
            violations.append("missing_year")
            break
    
    # Safety checks
    if not _no_new_claim_risk(original, candidate):
        violations.append("new_claim_risk")

    # Add personal salutation preservation
    personal_terms = re.findall(r'\b(twin|friend|fam|bro|sis|buddy)\b', original.lower())
    for term in personal_terms:
        if term not in candidate.lower():
            violations.append(f"missing_personal_term:{term}")
    
    return len(violations) == 0, violations

def quick_sentiment(text: str) -> str:
    """Quick sentiment heuristic for caching."""
    t = text.lower()
    if any(word in t for word in ["great", "amazing", "love", "awesome", "best", "excited"]):
        return "positive"
    elif any(word in t for word in ["bad", "terrible", "hate", "worst", "awful", "disappointed"]):
        return "negative"
    return "neutral"

def get_trending_hashtags() -> set:
    """Placeholder for trending hashtags API - replace with real implementation."""
    # TODO: implement real trending hashtags fetching
    return {"AI", "tech", "startup", "innovation", "productivity"}

# --- ADD: memoized light features (5–15 min stable) ---
@lru_cache(maxsize=8192)
def _light_features(text: str) -> dict:
    # very fast, re-used across prompt building & post-process heuristics
    s = quick_sentiment(text)
    tags = set(re.findall(r'#(\w+)', text.lower()))
    return {"sentiment": s, "hashtags": tags, "length": len(text)}

# --- ADD: trending cache (TTL) ---
TREND_TTL_SEC = int(os.getenv("TREND_TTL_SEC", "900"))  # 15 minutes
_TREND_CACHE = {"ts": 0.0, "data": set()}

def get_trending_cached() -> set:
    now = time.time()
    if now - _TREND_CACHE["ts"] > TREND_TTL_SEC:
        try:
            data = get_trending_hashtags()  # removed _SCORING_STACK_AVAILABLE gate
        except Exception:
            data = set()
        _TREND_CACHE.update({"ts": now, "data": set(map(str.lower, data))})
    return _TREND_CACHE["data"]

def _ensure_entities(analysis, text: str):
    """Use analyzer entities if present; fallback to simple regex extraction."""
    if getattr(analysis, "entities", None):
        return analysis.entities
    links    = re.findall(r'https?://\S+', text)
    mentions = re.findall(r'@\w+', text)
    hashtags = re.findall(r'#\w+', text)
    # If you don't have an Entities dataclass, swap this for a dict with same fields.
    return Entities(links=links, mentions=mentions, hashtags=hashtags)

def _strip_quote_bullets(s: str) -> str:
    lines = s.splitlines()
    cleaned = []
    for ln in lines:
        ln = re.sub(r'^\s*(>{1,3}\s*|\-\s+|\*\s+|\d+\.\s+)', '', ln)  # >, >>, -, *, 1.
        cleaned.append(ln.strip())
    return ' '.join(l for l in cleaned if l)

def precheck_and_parse(tweet_text: str):
    t = (tweet_text or "").strip()
    t = _strip_quote_bullets(t)
    if len(t) < 8:
        raise ValueError("tweet_too_short")
    lang = _detect_language_heuristic(t)
    analysis = TweetAnalyzer.analyze(t)
    analysis.language = lang
    analysis.entities = _ensure_entities(analysis, t)
    return t, analysis


# --- Real scoring stack selection (HTTP-first, then local persona_engine) ---
_real_run_elo_tournament = None
ENGINE = None

if SCORER_URL:
    _SCORING_STACK_AVAILABLE = True
    log.info("Using HTTP scorer at %s", SCORER_URL)
else:
    try:
        from persona_engine import EnhancedPersonaEngine, run_elo_tournament as _real_run_elo_tournament
        ENGINE = EnhancedPersonaEngine()  # or EnhancedPersonaEngine.from_config("config.json") if you have that
        _SCORING_STACK_AVAILABLE = True
        log.info("Using local persona_engine scorer.")
    except Exception as e:
        _SCORING_STACK_AVAILABLE = False
        log.warning("persona_engine not available; using fallback heuristic scorer. (%s)", e)


# Use persona_engine's loader to ensure correct Persona structure
from persona_engine import load_personas

# Wrapper that calls the real Elo if present, else falls back
def run_elo_tournament(texts: List[str], personas: List[Dict]) -> List[Dict]:
    """
    Returns a list of dicts like: [{"text": ..., "elo_score": float}, ...] ordered best->worst.
    """
    # Prefer HTTP Elo API if provided
    if SCORER_ELO_URL:
        try:
            payload = {"texts": texts, "personas": personas}
            r = requests.post(SCORER_ELO_URL, json=payload, timeout=SCORER_TIMEOUT)
            r.raise_for_status()
            data = r.json() if hasattr(r, "json") else []
            # Validate minimal shape
            if isinstance(data, list) and data and "text" in data[0]:
                return data
            log.warning("Elo API returned unexpected shape; falling back.")
        except Exception as e:
            log.warning("Elo API error: %s; falling back.", e)

    # Else use local function if present
    if _real_run_elo_tournament is not None:
        return _real_run_elo_tournament(texts, personas)

    # Last-resort fallback: sort by ML score
    sc = MLScorer()
    ranked = sorted(texts, key=lambda t: sc.score(t).composite, reverse=True)
    return [{"text": t, "elo_score": 1000 + (len(ranked) - i - 1) * 10} for i, t in enumerate(ranked)]

# =========================
# 1) DATA MODELS
# =========================
@dataclass
class Entities:
    links: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_text(cls, text: str) -> "Entities":
        """Extract entities from text using regex patterns."""
        links = LINK_RE.findall(text)
        mentions = MENTION_RE.findall(text)
        hashtags = HASHTAG_RE.findall(text)
        return cls(links=links, mentions=mentions, hashtags=hashtags)

@dataclass
class Analysis:
    tweet_type: str
    has_link: bool
    has_question: bool
    language: str
    entities: Entities

@dataclass
class Variant:
    text: str
    meta: Dict = field(default_factory=dict)

@dataclass
class Score:
    likes: float = 0.0
    retweets: float = 0.0
    replies: float = 0.0
    composite: float = 0.0
    details: Dict = field(default_factory=dict)

@dataclass
class RankedVariant:
    variant: Variant
    score: Score

# =========================
# 2) TWEET ANALYZER
# =========================
class TweetAnalyzer:
    @staticmethod
    def detect_type(text: str, ent: Entities) -> str:
        t = text.lower()
        if "breaking" in t or re.search(r'\b\d{4}\b', t) or ent.links:
            return "news"
        if any(x in t for x in ["free", "sale", "signup", "join", "limited", "waitlist"]):
            return "promo"
        if any(x in t for x in ["i ", "my ", "we ", "our ", "story", "learned", "thread"]):
            return "personal"
        return "general"

    @staticmethod
    def detect_language(text: str) -> str:
        # Use the same heuristic as elsewhere for consistency
        return _detect_language_heuristic(text)

    @classmethod
    def analyze(cls, text: str) -> Analysis:
        ent = extract_entities(text)
        return Analysis(
            tweet_type=cls.detect_type(text, ent),
            has_link=bool(ent.links),
            has_question=("?" in text),
            language=cls.detect_language(text),
            entities=ent,
        )

# --- REPLACE: PromptBuilder.build to use a single rich, numbered-output prompt ---
class PromptBuilder:
    @staticmethod
    def build(text: str, analysis, n: int = 30) -> str:
        kind = getattr(analysis, "tweet_type", None) or "general"
        cfg  = get_style_constraints(kind)
        ent  = analysis.entities
        lang_hint = "Write in the SAME language as the original."

        feat = _light_features(text)
        trending = list(get_trending_cached() or [])

        links    = getattr(ent, "links", []) or []
        mentions = getattr(ent, "mentions", []) or []
        keep_tags = getattr(ent, "hashtags", []) or []


        # --- Dynamic roles and length targets ---
        roles_bank = [
            "hook","question_specific","benefit_first","curiosity_gap","contrast","editor_choice",
            "myth_buster","imperative","proof","listicle","negative_hook","question_specific_2"
        ]
        variant_roles = {f"v{i+1}": roles_bank[i] for i in range(min(n, len(roles_bank)))}

        def _length_targets_for(n: int):
            m = {f"v{i}": "~100" for i in range(1, n+1)}
            if n >= 2: m["v2"] = "<=60"
            if n >= 3: m["v3"] = "200-240"
            return m

        # === Section A: Content + Length Rules ===
        content_rules = {
            "lead_with_best_bit": True,   # Start with the stat, claim, or payoff
            "sweet_spot_length_hint_chars": "~100",
            "length_targets_per_variant": _length_targets_for(n),
            "one_idea_only": True,  # “no multi-ideas; split or pick the strongest.”
            "prefer_concrete_over_filler": True,  # prefer concrete nouns/verbs over filler
            "require_takeaway_before_link": True, # include at least one concrete takeaway BEFORE any link
            "end_strong": True  # end with a punchy question, payoff, or crisp phrase
        }

        hard_constraints = {
            "preserve_facts": True,
            "preserve_language": True,
            "char_limit": CHAR_LIMIT,
            "keep_entities_verbatim": {
                "links": links,
                "mentions": mentions,
                "hashtags_to_keep": keep_tags
            },
            "emojis_max": int(cfg["max_emojis"]),
            "new_hashtags_max": int(cfg["max_new_hashtags"]),
            "cta_question_limit": 1,
            "language_hint": lang_hint
        }

        diversity_requirements = [
            "Each variant must be meaningfully different (no near-synonyms).",
            "At least one uses a specific, on-topic question (not generic).",
            "At least one is benefit-first.",
            "At least one uses a concise hook.",
            "At least one is an ultra-concise rewrite (<=60 chars).",
            "At least one is longer with substance (200–240 chars)."
        ]

        style_goals = {
            "type": kind,
            "notes": [
                "news: fact/stat up front; authoritative; optional topical hashtag.",
                "personal: vivid opener; one natural question allowed.",
                "promo: benefit-first; one proof/number; single crisp CTA.",
                "general: sharp hook; optional contrarian/novel angle."
            ]
        }

        # Output contract: strict JSON only
        output_schema = {
            "return": "JSON",
            "variants_key": "variants",
            "variants_count": n,
            "item_type": "string",
            "no_explanations": True
        }

        # Tight, tone-only few-shot (don’t copy wording)
        fewshot = [
            {
                "type": "news",
                "original": "New study shows remote work boosts productivity by 47%: https://x.y/z",
                "better":   "47% productivity boost, new study finds — details: https://x.y/z"
            },
            {
                "type": "personal",
                "original": "I learned to code late and it changed my career.",
                "better":   "I started coding late — it rewired my career. What skill did that for you?"
            },
            {
                "type": "promo",
                "original": "Try our tool for faster marketing.",
                "better":   "Save ~3 hours/week on campaigns with <tool>. Used by 1,200+ teams. Start free."
            },
            {
                "type": "general",
                "original": "Most advice is the same.",
                "better":   "Unpopular truth: 'universal' advice rarely fits your context — here's the swap that works."
            }
        ]

        payload = {
            "role": "expert_social_editor",
            "task": "Improve the tweet for engagement WITHOUT changing its factual meaning.",
            "original_tweet": text.strip(),
            "analysis": {
                "detected_type": kind,
                "features": feat,
                "trending_candidates": trending
            },
            "hard_constraints": hard_constraints,
            "content_rules": content_rules,
            "diversity_requirements": diversity_requirements,
            "style_goals": style_goals,
            "fewshot_for_tone_only": fewshot,
            "output_schema": output_schema
        }

        # --- ADD Step-B: Engagement Mechanics ---
        engagement_mechanics = {
            "variant_roles": {
                "v1": "hook",               # bold hook opener
                "v2": "question_specific",  # non-generic, on-topic question
                "v3": "benefit_first",      # reader payoff up front
                "v4": "curiosity_gap",      # tease + reveal framing
                "v5": "contrast",           # X vs Y / Before–After
                "v6": "editor_choice"       # free slot for diversity
            },
            "hook_requirements": {
                "must_begin_with": [
                    "Unpopular opinion:", "Reminder:", "Hot take:", "PSA:", "Heads up:",
                    "Fact:", "Myth:", "Reality:", "New:", "Data:", "Update:"
                ],
                "also_ok_if_starts_with_number": True
            },
            "question_rules": {
                "ban_generic": [
                    "Thoughts?", "What do you think?", "Agree?", "Opinions?",
                    "What are your thoughts?", "Your thoughts?"
                ],
                "ask_for_something_specific": True,
                "examples": [
                    "What would you change first about <topic>?",
                    "Where does this break in real teams?",
                    "What’s the hidden cost most people miss here?"
                ]
            },
            "benefit_first_rules": {
                "allowed_prefixes": ["Payoff:", "The win:", "You get:", "Outcome:", "What you’ll get:"],
                "must_state_reader_payoff_first": True
            },
            "curiosity_gap_rules": {
                "allowed_prefixes": ["Ever wonder why", "Here’s the twist:", "The catch:", "Counterintuitive bit:"],
                "structure_hint": "tease → short reveal"
            },
            "contrast_rules": {
                "allowed_patterns": ["X vs. Y", "Before → After", "Old way → New way", "Common advice vs. What actually helps"],
                "must_be_scannable": True
            },
            "patterned_openers_bank": [
                "The secret to <topic>…",
                "If you do <habit>, stop.",
                "Most people do X; the ones who win do Y.",
                "Here’s the 10-sec version:",
                "Steal this framework:"
            ]
        }
        payload["engagement_mechanics"] = engagement_mechanics

        # Final guard: force JSON-only response
        payload["assistant_instructions"] = (
            "Return ONLY a JSON object with a single key 'variants' whose value is an array of EXACTLY "
            f"{n} strings. No numbering, no commentary, no code fences."
        )

        # Gemini receives this JSON string as the prompt
        # --- FIX: convert any set in payload to list for JSON serialization ---
        def convert_sets(obj):
            if isinstance(obj, dict):
                return {k: convert_sets(v) for k, v in obj.items()}
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, list):
                return [convert_sets(i) for i in obj]
            else:
                return obj
        payload = convert_sets(payload)
        return json.dumps(payload, ensure_ascii=False)

## =========================
# 4) GEMINI CLIENT (real)
# =========================
class GeminiClient:
    def __init__(self, api_key: str, model: str, temperature: float, timeout: int = 20, retries: int = 2):
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")
        self._genai = genai
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model_name)

    def generate_variants(self, prompt: str, n: int, temperature: Optional[float] = None) -> str:
        """
        Generate variants using Gemini. Robustly handle finish_reason 2 (model stopped), empty, or malformed responses.
        Log finish_reason and error details. Fallback to deterministic variants if no valid output is found.
        """
        import logging
        actual_temp = temperature if temperature is not None else self.temperature
        last_err = None
        max_output_tokens = min(4096, 128 + n * 80)
        for attempt in range(max(1, self.retries)):
            try:
                resp = self._model.generate_content(
                    prompt,
                    generation_config=self._genai.GenerationConfig(
                        temperature=actual_temp,
                        max_output_tokens=max_output_tokens,
                    )
                )
                # Prefer .text if present and non-empty
                if hasattr(resp, "text") and resp.text:
                    return resp.text
                # Fallback: try to extract from candidates
                candidates = getattr(resp, "candidates", None)
                if candidates and isinstance(candidates, list):
                    for cand in candidates:
                        finish_reason = getattr(cand, "finish_reason", None)
                        # finish_reason 2 = STOP, 1 = SAFETY, 0 = FINISHED
                        if finish_reason not in (0, None):
                            logging.warning(f"[Gemini] LLM generation finish_reason={finish_reason} (not finished). Candidate: {getattr(cand, 'content', None)}")
                        content = getattr(cand, "content", None)
                        # Try .parts (list of Part objects with .text)
                        if content and hasattr(content, "parts"):
                            parts = [getattr(p, "text", "") for p in content.parts if getattr(p, "text", None)]
                            if parts:
                                return "\n".join(parts)
                        # Try .text directly
                        if hasattr(content, "text") and content.text:
                            return content.text
                    # If all candidates exhausted, log and break
                    logging.error(f"[Gemini] No valid candidates in LLM response. Candidates: {candidates}")
                else:
                    logging.error(f"[Gemini] LLM response missing candidates: {resp}")
                # If we reach here, no valid output found
                raise RuntimeError("Empty or invalid Gemini response (no usable text/candidates)")
            except Exception as e:
                last_err = e
                logging.warning(f"[Gemini] LLM generation attempt {attempt+1} failed: {e}")
                time.sleep(0.4)
        # All retries failed; log and raise
        logging.error(f"[Gemini] All LLM generation attempts failed. Last error: {last_err}")
        raise last_err


# =========================
# 5) POST-PROCESSING
# =========================
class PostProcessor:
    @staticmethod
    def normalize_spaces(s: str) -> str:
        # Collapse spaces/tabs, but keep bullets/newlines
        s = re.sub(r'[ \t]+', ' ', s)
        s = re.sub(r'\n{3,}', '\n\n', s)
        return s.strip()

    @staticmethod
    def limit_emojis(text: str, max_emojis: int) -> str:
        """Limit emojis to max_emojis by removing extras."""
        if count_emojis(text) <= max_emojis:
            return text
        keep = max_emojis
        def _strip_extra(m):
            nonlocal keep
            if keep > 0:
                keep -= 1
                return m.group(0)
            return ""  # drop extra
        return EMOJI_RE.sub(_strip_extra, text)

    @staticmethod
    def enforce_limits(
        text: str,
        orig_entities: "Entities",
        max_new_hashtags: int = MAX_NEW_HASHTAGS,
        max_emojis: int = MAX_EMOJIS_GENERAL,
        hard_char_cap: int = CHAR_LIMIT
    ) -> str:
        # 1) ensure original links/mentions are present (reinsert if LLM dropped)
        out = text
        for link in orig_entities.links:
            if link and link not in out:
                out = f"{out} {link}"
        for m in orig_entities.mentions:
            if m and m not in out:
                out = f"{out} {m}"

        # 2) clamp new hashtags
        orig_tags = set(h.lower() for h in orig_entities.hashtags)
        found = HASHTAG_RE.findall(out)
        kept, new = [], []
        for h in found:
            if h.lower() in orig_tags or h.lower() in {t.lower() for t in kept}:
                kept.append(h)
            else:
                new.append(h)
        # keep only the first allowed new hashtag
        if len(new) > max_new_hashtags:
            new = new[:max_new_hashtags]
        rebuilt = []
        seen = set()
        for token in out.split():
            if token.startswith('#') and token.lower() not in orig_tags:
                # emit only if in allowed 'new'
                if token.lower() in {x.lower() for x in new} and token.lower() not in seen:
                    rebuilt.append(token); seen.add(token.lower())
                # else drop extra new hashtags
            else:
                rebuilt.append(token)
        out = ' '.join(rebuilt)

        # 3) clamp emojis by removing the extras (keep earliest)
        if count_emojis(out) > max_emojis:
            keep = max_emojis
            def _strip_extra(m):
                nonlocal keep
                if keep > 0:
                    keep -= 1
                    return m.group(0)
                return ""  # drop extra
            out = EMOJI_RE.sub(_strip_extra, out)

        # 4) hard 280-char cap (truncate at nearest boundary if needed)
        if unicode_len(out) > hard_char_cap:
            # try trimming at last punctuation/space before cap
            cut = out[:hard_char_cap]
            punct = max(cut.rfind('.'), cut.rfind('!'), cut.rfind('?'), cut.rfind(','), cut.rfind(' ')) 
            out = cut if punct < 40 else cut[:punct].rstrip()  # avoid chopping too early

        # --- Tiny post-process scrub for bracketed placeholders and orphaned 'I'm aiming for' ---
        out = re.sub(r'\[[^\[\]\n]{1,40}\]', '', out)  # drop bracketed placeholders
        out = re.sub(r"\bI'?m\s+aiming\s+for\s*[—–-]?\s*$", "", out, flags=re.I)  # remove orphaned clause

        return PostProcessor.normalize_spaces(out)

    @staticmethod
    def smart_truncate(text: str, limit: int = CHAR_LIMIT) -> str:
        """Smart truncation at word/punctuation boundaries."""
        return _smart_truncate(text, limit=limit)

    @staticmethod
    def sanitize(text: str, entities: Entities, original: Optional[str] = None) -> str:
        out = text.strip()
        for m in entities.mentions:
            if m and m not in out:
                out = f"{m} {out}"
        for link in entities.links:
            if link and link not in out:
                out = f"{out} {link}"
        # Reinsert personal terms from ORIGINAL (not candidate)
        if original:
            for term in re.findall(r"\b(twin|friend|fam|bro|sis|buddy)\b", original.lower()):
                if term not in out.lower() and len(out) + len(term) + 1 <= CHAR_LIMIT:
                    out = f"{out} {term}"
        return out

    @staticmethod
    def ensure_takeaway_before_link(original: str, text: str) -> str:
        """
        If the tweet begins with a link or has <3 non-link tokens before the first link,
        move the link to the end with a neutral lead-in, preserving meaning.
        """
        tokens = text.strip().split()
        if not tokens:
            return text

        first_link_idx = None
        for i, tok in enumerate(tokens):
            if LINK_RE.match(tok):
                first_link_idx = i
                break

        if first_link_idx is None:
            return text

        # Count non-link words before the first link
        non_link_before = [t for t in tokens[:first_link_idx] if not LINK_RE.match(t)]
        if first_link_idx == 0 or len(non_link_before) < 3:
            link = tokens[first_link_idx]
            # Remove the first link occurrence
            remaining = tokens[:first_link_idx] + tokens[first_link_idx+1:]
            # Neutral, low-risk lead-in that doesn't add facts
            base = " ".join(remaining).strip()
            if not base:
                # fallback: pull a minimal neutral phrase from original if needed
                base = re.sub(r'\s+', ' ', original).strip()
            candidate = f"{base} — details: {link}".strip()
            return PostProcessor.normalize_spaces(candidate)
        return text

    @staticmethod
    def dedupe(texts: List[str]) -> List[str]:
        seen = set()
        out = []
        for t in texts:
            key = re.sub(r'\W+', '', t.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out

    @staticmethod
    def diversify(texts: List[str]) -> List[str]:
        # ensure some pattern diversity (question, stat, imperative). Minimal for skeleton.
        return texts  # No generic question injection; rely on ensure_engagement_mechanics()


# --- ADD: Ensure Step B roles exist across the set ---
# --- Tiny cleanup: dedupe links ---
def _dedupe_links(text: str) -> str:
    out, seen = [], set()
    for tok in text.split():
        if LINK_RE.match(tok):
            if tok in seen:
                continue
            seen.add(tok)
        out.append(tok)
    return " ".join(out)

def ensure_engagement_mechanics(variants: List[str], original: str, ent: Entities) -> List[str]:
    import re
    # Don’t auto-hook list/how-to posts
    if '\n' in original or re.search(r'(?i)\bhow to\b', original):
        return variants  # skip hook/question/contrast injection for checklists

    out = list(variants)

    have_hook      = any(is_hook_variant(v)               for v in out)
    have_question  = any(is_specific_question(v, original) for v in out)
    have_benefit   = any(is_benefit_first(v)              for v in out)
    have_curiosity = any(has_curiosity_gap(v)             for v in out)
    have_contrast  = any(has_contrast(v)                  for v in out)

    def safe_prepend(idx: int, prefix: str):
        s = out[idx]
        candidate = f"{prefix} {s}"
        candidate = PostProcessor.ensure_takeaway_before_link(original, candidate)
        candidate = reinsert_missing_entities(candidate, ent)
        candidate = PostProcessor.enforce_limits(candidate, ent, MAX_NEW_HASHTAGS, MAX_EMOJIS_GENERAL, CHAR_LIMIT)
        candidate = _dedupe_links(candidate)
        out[idx] = candidate

    if not out:
        return out

    shortest_i = min(range(len(out)), key=lambda i: len(out[i]))
    longest_i  = max(range(len(out)), key=lambda i: len(out[i]))

    if not have_hook:
        safe_prepend(shortest_i, "Reminder:")

    if not have_question:
        # Build a sane question topic
        m_pair = re.search(r'\b\d+\s+\w+', original)
        topic = (ent.hashtags[0] if ent.hashtags else ent.mentions[0] if ent.mentions else (m_pair.group(0) if m_pair else "this process"))
        q = f"What would you change first about {topic}?"
        s = out[0]
        cand = (s.rstrip(".! ") + " " + q).strip()
        cand = PostProcessor.enforce_limits(cand, ent, MAX_NEW_HASHTAGS, MAX_EMOJIS_GENERAL, CHAR_LIMIT)
        cand = _dedupe_links(cand)
        out[0] = cand

    if not have_benefit:
        safe_prepend(shortest_i, "You get:")

    if not have_curiosity:
        safe_prepend(longest_i, "Ever wonder why? Here’s the twist:")

    if not have_contrast:
        safe_prepend(longest_i, "Before → After:")

    return out

# --- Type-specific style enforcement + one-idea validator --------------------
# Put these near your other helpers (e.g., after ensure_engagement_mechanics)

NEWS_ALLOWED_PREFIXES = ("New:", "Data:", "Update:", "Fact:", "Breaking:")
BENEFIT_VERBS = {
    "save","cut","boost","increase","reduce","improve","grow","speed","ship","get","win"
}
CTA_RE = re.compile(
    r'\b(join|sign\s*up|get\s*started|try|download|subscribe|enroll|book|apply|start\s*(free|now)|buy|shop)\b',
    re.IGNORECASE
)
HEDGE_RE = re.compile(
    r'\b(kinda|sort of|maybe|perhaps|I think|I feel|seems|might|could|probably|IMO|IMHO)\b',
    re.IGNORECASE
)
WEAK_OPENERS_RE = re.compile(
    r'^\s*(today|some|this|there|here|just|FYI|announcing|sharing|random)\b[:,\s-]*',
    re.IGNORECASE
)

def _first_sentence_chunks(s: str) -> Tuple[str, str]:
    """Return (first_sentence, rest) using light punctuation split."""
    parts = re.split(r'(?<=[.!?])\s+', s.strip(), maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]

def _starts_with_stat_or_prefix(s: str) -> bool:
    t = s.strip()
    if any(t.startswith(p) for p in NEWS_ALLOWED_PREFIXES): 
        return True
    if KEY_NUMBER_RE.match(t) or YEAR_RE.match(t): 
        return True
    return False

def _move_first_numeric_clause_front(s: str) -> str:
    """If a number/year exists later, bring that sentence to the front."""
    sentences = re.split(r'(?<=[.!?])\s+', s.strip())
    idx = None
    for i, sen in enumerate(sentences):
        if KEY_NUMBER_RE.search(sen) or YEAR_RE.search(sen):
            idx = i; break
    if idx is None or idx == 0: 
        return s
    ordered = [sentences[idx]] + sentences[:idx] + sentences[idx+1:]
    return PostProcessor.normalize_spaces(" ".join(ordered))

def _cap_questions(text: str, max_q: int) -> str:
    """Keep at most max_q question marks; convert extras to periods."""
    if max_q < 0: 
        return text
    out, seen = [], 0
    for ch in text:
        if ch == "?":
            if seen < max_q:
                out.append("?"); seen += 1
            else:
                out.append(".")
        else:
            out.append(ch)
    return PostProcessor.normalize_spaces("".join(out))

def _strip_hedging(text: str) -> str:
    return PostProcessor.normalize_spaces(HEDGE_RE.sub("", text))

def one_idea_only_ok(text: str) -> bool:
    """
    Reject variants that cram multiple ideas.
    Heuristics:
      - >2 sentences OR
      - Too many heavy separators (; • — /) OR
      - 2+ coordinating conjunctions (and/but/while/yet/whereas)
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len([s for s in sentences if s]) > 2:
        return False
    seps = len(re.findall(r'[;/•]|—|–| / ', text))
    if seps >= 2:
        return False
    conj = len(re.findall(r'\b(and|but|while|yet|whereas)\b', text, flags=re.IGNORECASE))
    return conj < 3

def compress_to_one_idea(text: str) -> str:
    """Keep just the first strong clause."""
    first, _rest = _first_sentence_chunks(text)
    # Also cut at long dashes/semicolons if present
    first = re.split(r'[;—–/•]', first)[0]
    return PostProcessor.smart_truncate(first, limit=CHAR_LIMIT)

def enforce_news_style(original: str, text: str) -> str:
    # No added questions if original had none
    if "?" not in original and "?" in text:
        text = text.replace("?", ".")
    # Lead with stat/fact/prefix
    if not _starts_with_stat_or_prefix(text):
        text = _move_first_numeric_clause_front(text)
        if not _starts_with_stat_or_prefix(text):
            text = f"Update: {text}"
    # Authoritative: remove hedging, avoid exclamations spam
    text = _strip_hedging(text)
    text = re.sub(r'!{2,}', '!', text)
    return text

def enforce_personal_style(original: str, text: str) -> str:
    # Keep max one question, but don't add first-person prefixes
    text = _cap_questions(text, max_q=1)
    # If opener is weak, just trim it (no new words)
    if WEAK_OPENERS_RE.match(text):
        text = WEAK_OPENERS_RE.sub('', text).strip().capitalize()
    return text

def _first_clause(text: str) -> str:
    return re.split(r'[.!?—–,:;]\s+', text.strip(), maxsplit=1)[0]

def _is_benefit_first(text: str) -> bool:
    first = _first_clause(text).lower()
    if any(first.startswith(p.lower()) for p in BENEFIT_PREFIXES): 
        return True
    return any(v in first.split() for v in BENEFIT_VERBS)

def _has_single_cta(text: str) -> bool:
    return len(CTA_RE.findall(text)) <= 1

def _keep_first_cta_drop_rest(text: str) -> str:
    """If multiple CTAs, keep the sentence containing the first and drop other CTA sentences."""
    if _has_single_cta(text): 
        return text
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    out, kept_first = [], False
    for s in sentences:
        if CTA_RE.search(s):
            if not kept_first:
                out.append(s); kept_first = True
            # else drop
        else:
            out.append(s)
    return PostProcessor.normalize_spaces(" ".join(out))

def enforce_promo_style(original: str, text: str) -> str:
    # Payoff before link (your helper already moves early links, but ensure benefit-first)
    if not _is_benefit_first(text):
        text = f"You get: {text}"
    # Require one proof/number if the original had one (don't fabricate)
    if (KEY_NUMBER_RE.search(original) or YEAR_RE.search(original)) and not (KEY_NUMBER_RE.search(text) or YEAR_RE.search(text)):
        # move numeric sentence from original if missing
        text = PostProcessor.normalize_spaces(f"{_first_clause(original)} — {text}")
    # Exactly one CTA
    text = _keep_first_cta_drop_rest(text)
    return text

def enforce_general_style(original: str, text: str) -> str:
    # Ensure there is a hook/novel angle; if not, prepend an allowed hook
    if not is_hook_variant(text):
        # Prefer "Hot take:" if it won't clash with tone
        text = f"Hot take: {text}"
    return text

def apply_type_style(original: str, text: str, kind: str) -> Tuple[str, bool]:
    """
    Returns (possibly_edited_text, ok_flag). ok_flag False means it violates 'one idea only'.
    """
    if kind == "news":
        text = enforce_news_style(original, text)
    elif kind == "personal":
        text = enforce_personal_style(original, text)
    elif kind == "promo":
        text = enforce_promo_style(original, text)
    else:  # general
        text = enforce_general_style(original, text)

    # One-idea-only gate
    if not one_idea_only_ok(text):
        text = compress_to_one_idea(text)
        if not one_idea_only_ok(text):
            return text, False  # still too multi-idea; caller may drop
    return text, True


# =========================
# 6) SCORERS (stubs)
# =========================
def _normalize_engine_output(data):
    """Return (likes, retweets, replies) as floats from various engine output shapes."""
    # pydantic/dataclass/obj → dict
    if hasattr(data, "dict"):
        data = data.dict()
    if isinstance(data, dict):
        likes = float(data.get("likes", 0.0))
        retweets = float(data.get("retweets", data.get("rts", 0.0)))
        replies = float(data.get("replies", 0.0))
        return (likes, retweets, replies)
    raise TypeError(f"Unsupported engine output shape: {type(data)} {repr(data)[:120]}")

class BaseScorer:
    def score(self, text: str) -> Score:
        raise NotImplementedError

class MLScorer(BaseScorer):
    _scorer_warmed_up = False
    _scorer_warmup_lock = threading.Lock()

    def __init__(self, alpha: float = None, beta: float = None, followers: Optional[int] = None):
        # TODO: wire your EnhancedPersonaEngine.predict_virality here
        self.ready = _SCORING_STACK_AVAILABLE
        self.alpha = alpha or ALPHA_RETWEETS
        self.beta = beta or BETA_REPLIES
        self.followers = followers

        # Warm up scorer only once per process
        if SCORER_URL and not MLScorer._scorer_warmed_up:
            with MLScorer._scorer_warmup_lock:
                if not MLScorer._scorer_warmed_up:
                    wait_for_scorer_ready(timeout=15, interval=0.5, verbose=True)
                    MLScorer._scorer_warmed_up = True

    def score(self, text: str, followers: Optional[int] = None) -> Score:
        use_followers = followers if followers is not None else self.followers or globals().get("FOLLOWERS_FOR_RUN", None)
        # 1) HTTP scorer takes precedence if set
        if SCORER_URL:
            payload = {"text": text}
            if use_followers is not None:
                payload["followers"] = int(use_followers)
            last_err = None
            for _ in range(2):
                try:
                    r = _SESSION.post(SCORER_URL, json=payload, timeout=SCORER_TIMEOUT)
                    r.raise_for_status()
                    data = r.json()
                    likes = float(data.get("likes", 0.0))
                    rts   = float(data.get("retweets", data.get("rts", 0.0)))
                    reps  = float(data.get("replies", 0.0))
                    comp  = likes + self.alpha * rts + self.beta * reps
                    return Score(likes=likes, retweets=rts, replies=reps, composite=comp, details={"raw": data})
                except Exception as e:
                    last_err = e
                    log.warning("Scorer API error: %s", e)
                    time.sleep(0.25)
            log.error("Scorer API failed after retries: %s. Falling back.", last_err)

        # 2) Local persona_engine (your repo)
        if ENGINE is not None:
            try:
                for method_name in ("predict_virality", "predict", "score"):
                    fn = getattr(ENGINE, method_name, None)
                    if not callable(fn):
                        continue
                    raw = fn(text)
                    # helpful once: log the shape
                    log.debug("persona_engine output type=%s preview=%r", type(raw), raw if isinstance(raw, (int,float)) else (list(raw.keys()) if isinstance(raw, dict) else raw))
                    likes, rts, reps = _normalize_engine_output(raw)
                    comp = likes + self.alpha * rts + self.beta * reps
                    return Score(likes=likes, retweets=rts, replies=reps, composite=comp, details={"raw": raw})
            except Exception as e:
                log.warning("persona_engine scoring failed; using heuristic fallback. (%s)", e)

        # 3) Final fallback heuristic
        hook_bonus = 0.2 if any(x in text.lower() for x in ["breaking", "new", "just in", "thread"]) else 0.0
        q_bonus    = 0.15 if "?" in text else 0.0
        len_bonus  = 0.2  if 80 <= len(text) <= 180 else 0.0
        comp = 1.0 + hook_bonus + q_bonus + len_bonus
        return Score(likes=comp, retweets=comp*0.35, replies=comp*0.25, composite=comp)

class EloScorer(BaseScorer):
    def __init__(self):
        self.ready = _SCORING_STACK_AVAILABLE

    def score_many(self, variants: List[str]) -> List[Score]:
        if not self.ready:
            # simple per-text heuristic for skeleton
            return [MLScorer().score(v) for v in variants]
        # TODO: call run_elo_tournament([...]) and map back to Score per variant
        # return [...]
        return [Score(composite=1.0) for _ in variants]

# --- ADD: deterministic variants (fallback if LLM fails) ---
import hashlib
RNG_SALT = "improve.py-fallback-v1"

def _seed_from(text: str) -> int:
    h = hashlib.sha256((RNG_SALT + text).encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def deterministic_variants(original: str, n: int) -> List[str]:
    rnd = random.Random(_seed_from(original))
    base = original.strip()

    personal_terms = re.findall(r'\b(twin|friend|fam|bro|sis|buddy)\b', original.lower())
    base_term = (personal_terms[0] + '.') if personal_terms else ""

    def add_cta(t):
        CTAS = [
            f"Stay positive {base_term}",
            f"Keep smiling {base_term}",
            f"Sending good vibes {base_term}",
            f"Thoughts {base_term}",
        ]
        cta = rnd.choice(CTAS)
        sep = " " if t and t[-1].isalnum() else ""
        return f"{t}{sep}{cta}"

    def stat_first(t):
        urls = LINK_RE.findall(t)
        safe = LINK_RE.sub("__URL__", t)
        # Improved regex: captures optional $/currency, keeps %/k/K, etc.
        m = re.search(r'(\$?\d[\d,\.\'%kK]*\b.*?)([.!?])', safe)
        if not m:
            return t
        chunk = m.group(1).replace("__URL__", urls[0] if urls else "")
        # avoid duplicating the same clause
        if chunk and chunk.lower() in t.lower():
            return t
        return (chunk.strip() + " — " + t).strip()

    def tighten(t):
        return tighten_to_limit(t, limit=260)

    def punchy_open(t):
        OPENERS = ["Hot take:", "Real talk:", "Quick win:", "Underrated:", "Reminder:"]
        return f"{rnd.choice(OPENERS)} {t}"

    def emoji_trim(t):
        return PostProcessor.limit_emojis(t, max_emojis=1)

    strategies = [add_cta, stat_first, tighten, punchy_open, emoji_trim]
    outs = set()
    for _ in range(max(2, n) * 2):
        t = base
        for _ in range(rnd.randint(1, 3)):
            t = rnd.choice(strategies)(t)
        outs.add(PostProcessor.sanitize(t, Entities.from_text(base), original=original))
    # ensure size n (dedupe later anyway)
    return list(outs)[:n]

# --- ADD: helpers for key-fact preservation and constraint filtering ---------

def _extract_key_facts(text: str) -> Dict[str, set]:
    """Pull out meaning-critical literal tokens we must preserve."""
    nums = set(KEY_NUMBER_RE.findall(text))
    # fix grouping: capture the full year token instead of the leading group
    years = set(re.findall(r'(?:19|20)\d{2}', text))
    return {"numbers": nums, "years": years}

def _passes_key_constraints(candidate: str, ent: Entities, key: Dict[str, set]) -> Tuple[bool, List[str]]:
    """Check that links/@mentions/numbers/years survive; return (ok, violated_rules[])"""
    violated = []
    for link in getattr(ent, "links", []):
        if link and link not in candidate:
            violated.append("missing_link"); break
    for m in getattr(ent, "mentions", []):
        if m and m not in candidate:
            violated.append("missing_mention"); break
    for n in key["numbers"]:
        if n and n not in candidate:
            violated.append("missing_number"); break
    for y in key["years"]:
        if y and y not in candidate:
            violated.append("missing_year"); break
    if len(candidate) > CHAR_LIMIT:
        violated.append("char_limit_exceeded")
    return (len(violated) == 0), violated

def _has_generic_cta(text: str) -> bool:
    generic_ctas = {
        "thoughts?", "what do you think?", "opinions?",
        "agree?", "comments?", "reactions?",
        "what are your thoughts?", "what would you add?"
    }
    t = (text or "").lower().replace("'", "").replace('"', '')
    return any(cta in t for cta in generic_ctas)

# --- ADD: tweet-type heuristic ---------------------------------------------

# --- ADD: per-type constraints + getter ------------------------------------

STYLE = {
    "news":     {"max_emojis": 1, "max_new_hashtags": 1, "cta_required": False, "question_allowed": False,
                 "notes": ["clarity", "authority", "stat_up_front"]},
    "personal": {"max_emojis": 2, "max_new_hashtags": 1, "cta_required": True,  "question_allowed": True,
                 "notes": ["vivid_opener", "invite_response"]},
    "promo":    {"max_emojis": 1, "max_new_hashtags": 1, "cta_required": True,  "question_allowed": False,
                 "notes": ["benefit_first", "proof_number", "single_cta"]},
    "general":  {"max_emojis": 1, "max_new_hashtags": 1, "cta_required": False, "question_allowed": True,
                 "notes": ["hook+novel_angle"]}
}
def get_style_constraints(kind: str):
    return STYLE.get(kind or "general", STYLE["general"])

# --- ADD: simple numbered list parser --------------------------------------

NUM_ITEM_RE = re.compile(r'^\s*(?:\d+[\).\s-])\s*(.+)$')
def parse_numbered_list(text: str, expected: int) -> List[str]:
    out = []
    for line in text.splitlines():
        m = NUM_ITEM_RE.match(line)
        if m:
            s = m.group(1).strip()
            if s: out.append(s)
    if not out:
        out = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return out[:expected]


# --- ADD: robust parser to handle JSON output or fallback to numbered lines ---
def parse_gemini_variants(response_text: str, expected: int) -> List[str]:
    if not isinstance(response_text, str):
        return []

    s = response_text.strip()

    # 1) If there’s a fenced block, pull JSON out of it
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if m:
        s = m.group(1).strip()

    # 2) Try to isolate the first {...} that contains "variants"
    if "variants" in s:
        try:
            start = s.index("{")
            end = s.rfind("}") + 1
            obj = json.loads(s[start:end])
            if isinstance(obj, dict) and isinstance(obj.get("variants"), list):
                return [str(x).strip() for x in obj["variants"]][:expected]
        except Exception:
            pass

    # 3) Last resort: numbered-line fallback
    return parse_numbered_list(response_text, expected=expected)

# --- ADD: tighten to <=260 chars while preserving entities & meaning ---

def tighten_to_limit(text: str, limit: int = 260, entities: "Entities" = None) -> str:
    if len(text) <= limit:
        return text
    entities = entities or Entities.from_text(text)
    # keep entities verbatim, shorten around them
    keep_tokens = list(entities.links) + list(entities.mentions) + list(entities.hashtags)
    # soft split by sentences/phrases
    parts = re.split(r'([.!?]\s+)', text)
    out = ""
    for i in range(0, len(parts), 2):
        chunk = (parts[i] + (parts[i+1] if i+1 < len(parts) else "")).strip()
        if not chunk:
            continue
        # prefer chunks that contain key entities or numbers
        if any(k in chunk for k in keep_tokens) or re.search(r'\d', chunk):
            cand = (out + " " + chunk).strip()
        else:
            # try to keep the hook + last claim
            cand = (out + " " + chunk).strip()
        if len(cand) <= limit:
            out = cand
        else:
            break
    # last-resort hard trim at word boundary
    if len(out) > limit:
        out = PostProcessor.smart_truncate(out, limit=limit)
    # always ensure <= 280 ultimately
    return out[:min(limit, CHAR_LIMIT)]

# --- ADD: smart truncate + reinsert entities + dedupe + diversity ----------

def _clean_tail(s: str) -> str:
    return re.sub(r'[\s,;:–—-]+$', '', s).strip()

def _smart_truncate(s: str, limit: int = CHAR_LIMIT) -> str:
    if len(s) <= limit: 
        return _clean_tail(s)
    cut = s[:limit]
    for p in [r'\.\s', r'!\s', r'\?\s', r'—\s', r',\s', r'\s']:
        m = list(re.finditer(p, cut))
        if m:
            return _clean_tail(cut[:m[-1].end()])
    return _clean_tail(cut)

def reinsert_missing_entities(candidate: str, ent) -> str:
    out = candidate
    for link in getattr(ent, 'links', []):
        if link and link not in out:
            # Insert after first sentence if possible to preserve flow
            try:
                sentences = re.split(r'(?<=[.!?])\s+', out)
                if len(sentences) > 1:
                    out = f"{sentences[0]} {link} {' '.join(sentences[1:])}"
                else:
                    out = f"{out} {link}"
            except Exception:
                out = f"{out} {link}"
            # ensure char limit
            if len(out) > CHAR_LIMIT:
                out = _smart_truncate(out, CHAR_LIMIT)
    for m in getattr(ent, 'mentions', []):
        if m and m not in out and len(out) + len(m) + 1 <= CHAR_LIMIT:
            out = (m + " " + out).strip()
    # Also reinsert original hashtags if missing
    for h in getattr(ent, 'hashtags', []):
        if h and h not in out and len(out) + len(h) + 1 <= CHAR_LIMIT:
            out = (out + " " + h).strip()
    return out


def enforce_length_variety(variants: List[str], original: str, entities: Entities) -> List[str]:
    """
    Ensure at least one <=60 chars and one 200–240 chars variant if feasible,
    without violating constraints. We only shorten (never fabricate content).
    """
    have_ultra = any(len(v) <= 60 for v in variants)
    have_long  = any(200 <= len(v) <= 240 for v in variants)

    out = list(variants)

    # Ultra-short: derive by aggressive tighten from the best available candidate
    if not have_ultra and variants:
        base = sorted(variants, key=len)[0]  # shortest as starting point
        shorty = PostProcessor.smart_truncate(base, limit=58)
        shorty = reinsert_missing_entities(shorty, entities)
        if len(shorty) <= 60:
            out[0] = shorty  # replace first slot

    # Long-ish: choose the longest under cap and relax truncation a bit (if >240, tighten)
    if not have_long and variants:
        # Prefer a candidate between 180..280 to tighten into 200–240
        cand = max(variants, key=len)
        if len(cand) > 240:
            cand2 = tighten_to_limit(cand, limit=235, entities=entities)
            out[-1] = cand2
        elif len(cand) < 200 and len(cand) > 140:
            # If it's close, keep as-is; we won't pad/invent text
            pass

    return out

def dedupe_near(lines: List[str], thresh: int = 92) -> List[str]:
    """Dedupe using fuzzy matching. For now, use simple heuristic since fuzzywuzzy not imported."""
    kept = []
    for s in lines:
        # Simple similarity check - count common words
        if not any(_simple_similarity(s, k) >= thresh for k in kept):
            kept.append(s)
    return kept

def _simple_similarity(a: str, b: str) -> int:
    """Simple word-based similarity as fallback for fuzzy matching."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return int((intersection / union) * 100) if union > 0 else 0

def ensure_diversity(cands: List[str]) -> List[str]:
    if not cands:
        return cands
    # Light nudge only (don’t add any generic question text)
    has_stat = any(KEY_NUMBER_RE.search(c) for c in cands)
    out = []
    import re
    for s in cands:
        # Filter obvious junk before keeping a variant
        if re.search(r'\babout\s+\d+\?\b', s):
            continue
        # Clean up 'Reminder:'/Hot take: prefix if doubled
        s = re.sub(r'\bReminder:\s*Hot take:\s*', 'Hot take: ', s)
        out.append(s)
    return out

# --- ADD: optional topical hashtag enrichment -------------------------------

def maybe_add_trending_hashtag(s: str, trending: Optional[set]) -> str:
    if not trending: return s
    existing = {h.lower() for h in re.findall(r'#\w+', s)}
    for tag in trending:
        token = f"#{tag}"
        if token.lower() in existing:
            return s
        if len(s) + len(token) + 1 <= CHAR_LIMIT:
            return (s + " " + token).strip()
    return s

# --- ADD: Persona sampling and scoring functions -------------------------------

PERSONA_SAMPLE_MAX = int(os.getenv("PERSONA_SAMPLE_MAX", "1000"))

def _sample_personas(personas, text: str, k: int = PERSONA_SAMPLE_MAX):
    if len(personas) <= k:
        return personas
    return _sample_personas_with_seed(personas, text, k)

def _sample_personas_with_seed(personas, text: str, k: int = PERSONA_SAMPLE_MAX, seed: Optional[int] = None):
    use_seed = seed if seed is not None else int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rnd = random.Random(use_seed)
    return rnd.sample(personas, k)

# --- ADD: light hook/clarity rubric for tie-breaks --------------------------

def quick_rubric(s: str) -> float:
    hook = 1.0 if re.match(r'^[""\'!?A-Z0-9#@]', s) else 0.0
    q    = 0.3 if '?' in s else 0.0
    num  = 0.2 if KEY_NUMBER_RE.search(s) else 0.0
    return hook + q + num

# --- ADD: safety filter used during post-process ----------------------------

def safety_ok(original: str, candidate: str, ent) -> Tuple[bool, str]:
    # no new mentions
    new_mentions = set(re.findall(r'@\w+', candidate)) - set(getattr(ent, 'mentions', []))
    if new_mentions: return False, "new_mentions"
    # no emails / phones
    if EMAIL_RE.search(candidate): return False, "pii_email"
    if PHONE_RE.search(candidate): return False, "pii_phone"
    return True, ""

# --- ADD: anonymized telemetry sink -----------------------------------------

def _log_telemetry(original: str, winner: RankedVariant, timings: Dict, analysis, base_score, selection_mode: str):
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "type": analysis.tweet_type,
        "selection_mode": selection_mode,
        "timings_ms": timings,
        "base_composite": getattr(base_score, "composite", None),
        "winner_len": len(winner.variant.text),
        "winner_composite": getattr(winner.score, "composite", None),
    }
    try:
        with open("improve_telemetry.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception as e:
        log.debug("telemetry write failed: %s", e)

# --- ADD: helpers for elo + blended scoring ---
from typing import List, Tuple

def rank_with_elo(original: str, texts: List[str], personas: List[Dict]):
    # Score original ML once for deltas
    base_score = MLScorer().score(original)
    # Run Elo tournament using your existing tournament function (add original too)
    pack = [original] + texts
    results = run_elo_tournament(pack, personas)  # you already have this
    # Map back into RankedVariant-like objects
    ranked = []
    for r in results:
        rv = RankedVariant(
            variant=Variant(text=r["text"]),
            score=Score(  # fill composite from Elo (normalize later if needed)
                likes=0.0, retweets=0.0, replies=0.0, composite=r.get("elo_score", r.get("elo", 1000.0))
            )
        )
        ranked.append(rv)
    # Ensure best is first; drop the original from alternates presentation later if needed
    ranked = sorted(ranked, key=lambda x: x.score.composite, reverse=True)
    return ranked, base_score

def blended_rank(original: str, variants: List[str], scorer_ml: "MLScorer", personas: List[Dict]):
    # 1) ML scores (batch original and variants)
    texts = [original] + variants
    ml_scores_all = _ml_score_parallel(texts, followers=scorer_ml.followers if hasattr(scorer_ml, 'followers') else None)
    ml_scores = {v: ml_scores_all[v] for v in variants}
    base_ns = ml_scores_all[original]
    base_ml = Score(likes=base_ns.likes, retweets=base_ns.retweets,
                    replies=base_ns.replies, composite=base_ns.composite)

    # 2) Elo scores (normalize 0..1) for original + variants
    results = run_elo_tournament([original] + variants, personas)
    if not results:
        elo_raw = {t: 0.0 for t in [original] + variants}
    else:
        elo_raw = {r["text"]: r.get("elo_score", r.get("elo", 1000.0)) for r in results}

    mn, mx = min(elo_raw.values()), max(elo_raw.values())
    def norm(x): return 0.0 if mx == mn else (x - mn) / (mx - mn)

    # 3) Build blended for variants
    blended = []
    for v in variants:
        ml_c  = ml_scores[v].composite
        elo_c = norm(elo_raw.get(v, mn))
        comp  = BLEND_W_ML * ml_c + BLEND_W_ELO * elo_c
        blended.append(
            RankedVariant(
                variant=Variant(text=v),
                score=Score(
                    likes=ml_scores[v].likes,
                    retweets=ml_scores[v].retweets,
                    replies=ml_scores[v].replies,
                    composite=comp
                )
            )
        )

    blended.sort(key=lambda r: r.score.composite, reverse=True)
    if not blended:
        raise RuntimeError("No valid variants to rank. All candidates were filtered out or scoring failed.")
    winner     = blended[0]
    # Dynamic alternates cap: respect --num_variants or uncap to all alternates
    import inspect
    alt_cap = max(0, int(len(variants) - 1))
    frame = inspect.currentframe().f_back
    if frame and "alt_cap" in frame.f_locals:
        alt_cap = frame.f_locals["alt_cap"]
    alternates = blended[1:1+max(0, int(alt_cap))]

    # 4) Compute a BLENDED base score for the ORIGINAL too (apples-to-apples)
    base_blended = Score(
        likes=base_ml.likes,
        retweets=base_ml.retweets,
        replies=base_ml.replies,
        composite=(BLEND_W_ML * base_ml.composite + BLEND_W_ELO * norm(elo_raw.get(original, mn))),
        details=getattr(base_ml, "details", {})
    )

    return winner, alternates, base_blended

# --- ADD: ML scoring helpers (uses your existing MLScorer) ---
def _ml_score_set(texts: List[str], scorer: "MLScorer") -> Dict[str, "Score"]:
    out = {}
    for t in texts:
        s = scorer.score(t)  # expects .likes, .retweets, .replies, .composite (we will override composite)
        comp = (s.likes + ALPHA_RETWEETS * s.retweets + BETA_REPLIES * s.replies)
        s = dataclasses.replace(s, composite=comp) if hasattr(dataclasses, "replace") else s  # ok if you already set composite
        # if you can't replace, set attribute directly:
        try:
            s.composite = comp
        except Exception:
            pass
        out[t] = s
    return out

def _minmax(values: Dict[str, float]) -> Dict[str, float]:
    if not values:
        return {}
    v = list(values.values())
    lo, hi = min(v), max(v)
    if hi <= lo:
        return {k: 0.5 for k in values}  # flat case
    rng = hi - lo
    return {k: (values[k] - lo) / rng for k in values}

# --- ADD: Elo helpers (subset personas + rank normalization) ---
def _elo_rank_scores(texts: List[str], personas: List[Dict]) -> Dict[str, float]:
    # Subset personas for speed (cap at ELO_PERSONA_CAP)
    # Use first text as seed for deterministic sampling
    text_for_seed = texts[0] if texts else "default"
    personas_subset = _sample_personas(personas, text_for_seed, k=ELO_PERSONA_CAP)

    # Use your existing run_elo_tournament; it expects variants list
    # Build minimal variants = the texts list (original + candidates)
    tournament = run_elo_tournament(texts, personas_subset)
    # tournament is sorted best->worst; assign rank scores (best=1.0, worst=0.0)
    n = len(tournament)
    if n <= 1:
        return {tournament[0]['text']: 1.0} if n == 1 else {}

    # linear rank-to-score
    scores = {}
    for rank, row in enumerate(tournament, start=1):  # 1..n
        scores[row['text']] = 1.0 - ((rank - 1) / (n - 1))
    return scores

# --- ADD: blend ML composite with Elo rank score ---
# --- ADD: ML scoring worker for ProcessPool ---
def _ml_score_worker(text: str):
    s = MLScorer().score(text)  # returns likes/retweets/replies/composite
    comp = (s.likes + ALPHA_RETWEETS * s.retweets + BETA_REPLIES * s.replies)  # unify composite
    return (text, float(s.likes), float(s.retweets), float(s.replies), float(comp))

def _ml_score_parallel(texts: List[str], followers: Optional[int] = None) -> Dict[str, SimpleNamespace]:
    # Only use /predict endpoint for ML scoring
    if not SCORER_URL:
        # Fallback: local scoring (rare in your setup)
        results = {}
        for t in texts:
            s = MLScorer().score(t, followers=followers)
            comp = s.likes + ALPHA_RETWEETS * s.retweets + BETA_REPLIES * s.replies
            results[t] = SimpleNamespace(likes=float(s.likes), retweets=float(s.retweets),
                                         replies=float(s.replies), composite=float(comp))
        return results

    # Use /predict for each text
    use_followers = followers if followers is not None else globals().get("FOLLOWERS_FOR_RUN", None)
    results = {}
    for t in texts:
        payload = {"text": t, "return_details": False}
        if use_followers is not None:
            payload["followers"] = int(use_followers)
        r = _SESSION.post(SCORER_URL, json=payload, timeout=SCORER_TIMEOUT)
        r.raise_for_status()
        d = r.json()
        likes  = float(d.get("likes", 0.0))
        rts    = float(d.get("retweets", 0.0))
        reps   = float(d.get("replies", 0.0))
        comp   = likes + ALPHA_RETWEETS * rts + BETA_REPLIES * reps
        results[t] = SimpleNamespace(likes=likes, retweets=rts, replies=reps, composite=comp)
    return results

# --- REPLACE: choose_winner with success-criteria aware selection ------------

def choose_winner(original: str, variants: List[Variant], *, scorer: Optional[BaseScorer] = None
                 ) -> Tuple[RankedVariant, List[RankedVariant], Score]:
    """
    Scores original + candidates; only promote a variant if it beats the original.
    Returns (winner, alternates_sorted, original_score).
    """
    scorer = scorer or MLScorer()

    # Batch original and candidates for scoring
    texts = [original] + [v.text for v in variants]
    parallel_scores = _ml_score_parallel(texts, followers=globals().get("FOLLOWERS_FOR_RUN", None))  # {text: SimpleNamespace(...)}
    base_ns = parallel_scores[original]
    base_score = Score(likes=base_ns.likes, retweets=base_ns.retweets,
                       replies=base_ns.replies, composite=base_ns.composite)

    ranked = sorted(
        [
            RankedVariant(
                variant=v,
                score=Score(
                    likes=parallel_scores[v.text].likes,
                    retweets=parallel_scores[v.text].retweets,
                    replies=parallel_scores[v.text].replies,
                    composite=parallel_scores[v.text].composite,
                ),
            )
            for v in variants
        ],
        key=lambda r: r.score.composite,
        reverse=True,
    )

    if not ranked or ranked[0].score.composite <= base_score.composite:
        keep = RankedVariant(variant=Variant(text=original), score=base_score)
        return keep, [], base_score

    # Apply conciseness bonus: if a candidate is >=10% shorter than original, boost composite
    try:
        orig_len = len(original)
        for r in ranked:
            if len(r.variant.text) < orig_len * 0.9:
                r.score.composite *= 1.15
    except Exception:
        pass

    # Dynamic alternates cap: respect --num_variants or uncap to all alternates
    import inspect
    alt_cap = max(0, int(len(variants) - 1))
    frame = inspect.currentframe().f_back
    if frame and "alt_cap" in frame.f_locals:
        alt_cap = frame.f_locals["alt_cap"]
    return ranked[0], ranked[1:1+max(0, int(alt_cap))], base_score

# --- REPLACE: improve_tweet to implement the exact I/O contract --------------

# --- UPDATE improve_tweet signature defaults from config ---
def improve_tweet(
    tweet_text: str,
    mode: Optional[str] = None,
    num_variants: int = NUM_VARIANTS_DEFAULT,
    return_all: bool = False,
    followers: Optional[int] = None
) -> Dict:
    # Use followers from argument, or fallback to global if set
    if followers is None:
        followers = globals().get("FOLLOWERS_FOR_RUN", None)
    t_total = time.time()
    timings: Dict[str, float] = {}

    # 1) Pre-check & parse
    t0 = time.time()
    tweet_text, analysis = precheck_and_parse(tweet_text)
    timings["precheck_ms"] = round((time.time() - t0) * 1000, 2)

    # 2) Type detection (override only if mode is not provided)
    t0 = time.time()
    if not mode or mode == "auto":
        analysis.tweet_type = TweetAnalyzer.detect_type(tweet_text, analysis.entities)
    else:
        analysis.tweet_type = mode
    timings["type_detect_ms"] = round((time.time() - t0) * 1000, 2)

    # 3) Prompt
    t0 = time.time()
    prompt = PromptBuilder.build(tweet_text, analysis, n=num_variants)
    timings["prompt_ms"] = round((time.time() - t0) * 1000, 2)

    # 4) Generate variants with Gemini
    # --- REPLACE: LLM call with robust fallback ---
    t0 = time.time()
    fallback_used = False
    raw_lines: List[str] = []
    try:
        llm = GeminiClient(
            api_key=GEMINI_API_KEY,
            model=GEMINI_MODEL,
            temperature=TEMPERATURE,
            timeout=GEN_TIMEOUT_S,
            retries=GEN_RETRIES
        )
        response_text = llm.generate_variants(prompt, n=num_variants)
        if isinstance(response_text, str):
            raw_lines = parse_gemini_variants(response_text, expected=num_variants)
        else:
            raw_lines = list(response_text)[:num_variants]

        if len(raw_lines) < num_variants:
            need = num_variants - len(raw_lines)
            raw_lines += deterministic_variants(tweet_text, n=need)
            raw_lines = MiniPost.dedupe(raw_lines)[:num_variants]
    except Exception as e:
        log.warning("LLM generation failed (%s). Falling back to deterministic variants.", e)
        raw_lines = deterministic_variants(tweet_text, n=num_variants)
        fallback_used = True
    timings["llm_ms"] = round((time.time() - t0) * 1000, 2)

    # 5) Post-process and convert to Variant objects
    max_emojis_budget = MAX_EMOJIS_NEWS if analysis.tweet_type in {"news","announcement"} else MAX_EMOJIS_GENERAL
    processed = MiniPost.batch(
        original=tweet_text,
        candidates=raw_lines,
        entities=analysis.entities,
        max_new_hashtags=MAX_NEW_HASHTAGS,
        max_emojis=max_emojis_budget,
        limit=CHAR_LIMIT
    )
    variants = [Variant(text=v) for v in processed]

# =========================
# 9) CLI/LOCAL TEST
# =========================
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    import argparse

    parser = argparse.ArgumentParser(description="Tweet improvement engine")
    parser.add_argument("tweet_text", type=str, help="The tweet text to improve")
    parser.add_argument("--return_all", action="store_true", help="Include alternates in the output")
    parser.add_argument("--mode", type=str, default="auto", help="Improvement mode")
    parser.add_argument("--num_variants", type=int, default=NUM_VARIANTS_DEFAULT, help="Number of variants")
    parser.add_argument("--print_all_llm", action="store_true", help="Print all LLM-generated tweets (raw, before post-processing)")
    parser.add_argument("--followers", type=int, help="Follower count to use for scoring")
    parser.add_argument("--persona-seed", type=int, help="Seed for persona sampling (stabilizes Elo)")
    args = parser.parse_args()

    # --- Follower count: ask every run or via flag ---
    FOLLOWERS_FOR_RUN = args.followers
    if FOLLOWERS_FOR_RUN is None:
        try:
            FOLLOWERS_FOR_RUN = int(input("How many followers should we assume for scoring? ").strip())
        except Exception:
            FOLLOWERS_FOR_RUN = 100_000  # reasonable default

    # Persona seed for stable Elo sampling
    PERSONA_SEED_FOR_RUN = args.persona_seed if args.persona_seed is not None else FOLLOWERS_FOR_RUN

    # --- BEGIN: Show pre/post-processing variants for CLI/debug ---
    t_total = time.time()
    tweet_text = args.tweet_text
    mode = args.mode
    num_variants = args.num_variants
    return_all = args.return_all

    t0 = time.time()
    tweet_text_clean, analysis = precheck_and_parse(tweet_text)
    prompt = PromptBuilder.build(tweet_text_clean, analysis, n=num_variants)

    fallback_used = False
    raw_lines = []
    try:
        llm = GeminiClient(
            api_key=GEMINI_API_KEY,
            model=GEMINI_MODEL,
            temperature=TEMPERATURE,
            timeout=GEN_TIMEOUT_S,
            retries=GEN_RETRIES
        )
        response_text = llm.generate_variants(prompt, n=num_variants)
        if isinstance(response_text, str):
            raw_lines = parse_gemini_variants(response_text, expected=num_variants)
        else:
            raw_lines = list(response_text)[:num_variants]
        if len(raw_lines) < num_variants:
            need = num_variants - len(raw_lines)
            raw_lines += deterministic_variants(tweet_text_clean, n=need)
            raw_lines = MiniPost.dedupe(raw_lines)[:num_variants]
    except Exception as e:
        log.warning("LLM generation failed (%s). Falling back to deterministic variants.", e)
        raw_lines = deterministic_variants(tweet_text_clean, n=num_variants)
        fallback_used = True

    if args.print_all_llm:
        print("\n=== ALL LLM-GENERATED TWEETS (RAW) ===")
        for i, v in enumerate(raw_lines, 1):
            print(f"{i}. {v}")
    else:
        print("\n=== RAW VARIANTS (pre post-processing) ===")
        for i, v in enumerate(raw_lines, 1):
            print(f"{i}. {v}")

    orig_entities = analysis.entities
    max_emojis_budget = MAX_EMOJIS_NEWS if analysis.tweet_type in {"news","announcement"} else MAX_EMOJIS_GENERAL
    processed = MiniPost.batch(
        original=tweet_text_clean,
        candidates=raw_lines,
        entities=orig_entities,
        max_new_hashtags=MAX_NEW_HASHTAGS,
        max_emojis=max_emojis_budget,
        limit=CHAR_LIMIT
    )

    print("\n=== POST-PROCESSED VARIANTS ===")
    for i, v in enumerate(processed, 1):
        print(f"{i}. {v}")

    # 5) Call improve_tweet for final scoring and output
    # Patch all Elo/blend scoring calls to use the persona seed
    def patched_sample_personas(personas, text, k=PERSONA_SAMPLE_SIZE, seed=None):
        return _sample_personas_with_seed(personas, text, k=k, seed=PERSONA_SEED_FOR_RUN)

    _sample_personas_orig = _sample_personas
    _sample_personas = patched_sample_personas

    result = improve_tweet(
        tweet_text=args.tweet_text,
        mode=args.mode,
        num_variants=args.num_variants,
        return_all=args.return_all,
        followers=FOLLOWERS_FOR_RUN,
    )

    # Restore original _sample_personas in case of further use
    _sample_personas = _sample_personas_orig

    print("\n=== WINNER ===")

    if not result or not result.get("winner"):
        print("No winning variant could be selected. All candidates may have been filtered out or failed scoring.")
    else:
        print(result["winner"]["text"])
        print("Predicted:", result["winner"]["predicted"])
        print("Delta vs original:", result["winner"]["delta_vs_original"])

    print("\n=== ALTERNATES ===")
    if not result or not result.get("alternates"):
        print("No alternates available.")
    else:
        for alt in result["alternates"]:
            print("-", alt["text"], alt["predicted"])

    print("\n=== CONTEXT ===")
    if not result or not result.get("explanations"):
        print("No context/explanations available.")
    else:
        print("Why this won:", result["explanations"].get("why_this_won", "N/A"))
        print("Applied transforms:", result["explanations"].get("applied_transformations", "N/A"))
        print("Guardrails:", result.get("guardrails", "N/A"))
