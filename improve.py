# improve.py — skeleton (v0)
# Purpose: take a tweet, generate 5–7 improved variants with Gemini, score them
# with your prediction stack/personas, and return the best + context.
#
# Environment Variable Examples (override config.json):
# export SCORING_MODE=blend
# export NUM_VARIANTS=7
# export PERSONA_SAMPLE_SIZE=1200
# export GEMINI_MODEL="models/gemini-1.5-pro"
# export TEMPERATURE=0.9
# export MAX_NEW_HASHTAGS=1
# export MAX_EMOJIS=2
# export ALLOW_CTA=true

from __future__ import annotations
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

# optional HTTP scorer settings (useful when running a separate scorer service)
import requests  # <-- add this

SCORER_URL = os.getenv("SCORER_URL")  # e.g., http://localhost:8000/predict
SCORER_TIMEOUT = float(os.getenv("SCORER_TIMEOUT", "3"))

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
    NUM_VARIANTS: int = 6                      # 5–7 recommended
    SCORING_MODE: str = "blend"                # "ml" | "elo" | "blend"
    ALPHA_RETWEETS: float = 2.0
    BETA_REPLIES: float = 1.0
    MAX_NEW_HASHTAGS: int = 1
    MAX_EMOJIS: int = 2
    ALLOW_CTA: bool = True
    PERSONA_SAMPLE_SIZE: int = 1000            # if sub-sampling for Elo
    TRENDING_HASHTAGS_ENABLED: bool = True
    GEMINI_MODEL: str = "models/gemini-1.5-pro"
    TEMPERATURE: float = 0.85
    TIMEOUT: int = 20                          # seconds
    RETRIES: int = 2

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

    return Settings(**base)

# single global settings object
SETTINGS = load_settings()

# --- WIRE SETTINGS INTO EXISTING KNOBS ---
NUM_VARIANTS_DEFAULT   = SETTINGS.NUM_VARIANTS
SCORING_MODE_DEFAULT   = SETTINGS.SCORING_MODE         # "ml" | "elo" | "blend"
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

def precheck_and_parse(tweet_text: str):
    t = (tweet_text or "").strip()
    if len(t) < 8:
        raise ValueError("tweet_too_short")
    lang = _detect_language_heuristic(t)
    analysis = TweetAnalyzer.analyze(t)
    analysis.language = lang
    analysis.entities = _ensure_entities(analysis, t)
    return t, analysis

# --- Real scoring stack selection (HTTP-first) ---
_real_run_elo_tournament = None
if SCORER_URL:
    _SCORING_STACK_AVAILABLE = True
    log.info("Using HTTP scorer at %s", SCORER_URL)
else:
    try:
        from my_model import EnhancedPersonaEngine, run_elo_tournament as _real_run_elo_tournament
        _SCORING_STACK_AVAILABLE = True
        log.info("Using local my_model scorer.")
    except Exception as e:
        _SCORING_STACK_AVAILABLE = False
        log.warning("Scoring stack not imported yet. Using fallback pass-through scorer. (%s)", e)

# Helper function for loading personas
def load_personas(file_path: str) -> List[Dict]:
    """Load personas from JSON file, return empty list if not found."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        log.debug("Could not load personas from %s: %s", file_path, e)
        return []

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
    def build(text: str, analysis, n: int = 6) -> str:
        kind = getattr(analysis, "tweet_type", None) or "general"
        cfg  = get_style_constraints(kind)
        ent  = analysis.entities
        lang_hint = "Write in the SAME language as the original."

        feat = _light_features(text)
        trending = list(get_trending_cached() or [])

        links    = getattr(ent, "links", []) or []
        mentions = getattr(ent, "mentions", []) or []
        keep_tags = getattr(ent, "hashtags", []) or []

        # === Section A: Content + Length Rules ===
        content_rules = {
            # Lead with the best bit
            "lead_with_best_bit": True,   # Start with the stat, claim, or payoff

            # Hit the sweet spot length
            "sweet_spot_length_hint_chars": "~100",
            "length_targets_per_variant": {
                # ensure variety across the batch
                "v1": "~100",
                "v2": "<=60",     # ultra-short
                "v3": "200-240",  # longer, still crisp
                "v4": "~100",
                "v5": "~100",
                "v6": "~100"
            },

            # One idea per tweet
            "one_idea_only": True,  # “no multi-ideas; split or pick the strongest.”

            # Keep substance, trim fluff
            "prefer_concrete_over_filler": True,  # prefer concrete nouns/verbs over filler

            # Avoid over-linking
            "require_takeaway_before_link": True, # include at least one concrete takeaway BEFORE any link

            # End strong
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
        return json.dumps(payload, ensure_ascii=False)

## =========================
# 4) GEMINI CLIENT (real)
# =========================
class GeminiClient:
    def __init__(self, api_key: str, model: str, temperature: float, timeout: int = 20, retries: int = 2):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError:
            genai = None
        self._genai = genai
        self.api_key = api_key
        self.model_name = model
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        self._genai.configure(api_key=self.api_key)
        self._model = self._genai.GenerativeModel(self.model_name)

    def generate_variants(self, prompt: str, n: int, temperature: Optional[float] = None) -> str:
        # We return a RAW string; caller will parse numbered list.
        actual_temp = temperature if temperature is not None else self.temperature

        # Basic retry loop
        last_err = None
        for _ in range(max(1, self.retries)):
            try:
                resp = self._model.generate_content(
                    prompt,
                    generation_config=self._genai.GenerationConfig(
                        temperature=actual_temp,
                        max_output_tokens=1024,
                    )
                )
                # Some SDK versions use .text; others .candidates[0].content.parts
                if hasattr(resp, "text") and resp.text:
                    return resp.text
                # Fallback extraction
                if getattr(resp, "candidates", None):
                    parts = []
                    for p in getattr(resp.candidates[0], "content", {}).parts or []:
                        if getattr(p, "text", None):
                            parts.append(p.text)
                    if parts:
                        return "\n".join(parts)
                raise RuntimeError("Empty Gemini response")
            except Exception as e:
                last_err = e
                time.sleep(0.4)
        raise last_err


# =========================
# 5) POST-PROCESSING
# =========================
class PostProcessor:
    @staticmethod
    def normalize_spaces(s: str) -> str:
        return re.sub(r'\s+', ' ', s).strip()

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
def ensure_engagement_mechanics(variants: List[str], original: str, ent: Entities) -> List[str]:
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
        out[idx] = candidate

    if not out: 
        return out

    shortest_i = min(range(len(out)), key=lambda i: len(out[i]))
    longest_i  = max(range(len(out)), key=lambda i: len(out[i]))

    if not have_hook:
        safe_prepend(shortest_i, "Reminder:")

    if not have_question:
        topic = ent.hashtags[0] if ent.hashtags else (ent.mentions[0] if ent.mentions else (KEY_NUMBER_RE.search(original).group(0) if KEY_NUMBER_RE.search(original) else "this"))
        q = f"What would you change first about {topic}?"
        s = out[0]
        cand = (s.rstrip(".! ") + " " + q).strip()
        cand = PostProcessor.enforce_limits(cand, ent, MAX_NEW_HASHTAGS, MAX_EMOJIS_GENERAL, CHAR_LIMIT)
        out[0] = cand

    if not have_benefit:
        safe_prepend(shortest_i, "You get:")

    if not have_curiosity:
        safe_prepend(longest_i, "Ever wonder why? Here’s the twist:")

    if not have_contrast:
        safe_prepend(longest_i, "Before → After:")

    return out

# =========================
# 6) SCORERS (stubs)
# =========================
class BaseScorer:
    def score(self, text: str) -> Score:
        raise NotImplementedError

class MLScorer(BaseScorer):
    def __init__(self, alpha: float = None, beta: float = None):
        # TODO: wire your EnhancedPersonaEngine.predict_virality here
        self.ready = _SCORING_STACK_AVAILABLE
        self.alpha = alpha or ALPHA_RETWEETS
        self.beta = beta or BETA_REPLIES

    def score(self, text: str) -> Score:
        if SCORER_URL:
            payload = {"text": text}
            last_err = None
            for _ in range(2):  # retry twice
                try:
                    r = requests.post(SCORER_URL, json=payload, timeout=SCORER_TIMEOUT)
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

        # fallback heuristic if API fails
        hook_bonus = 0.2 if any(x in text.lower() for x in ["breaking", "new", "just in", "thread"]) else 0.0
        q_bonus = 0.15 if "?" in text else 0.0
        len_bonus = 0.2 if 80 <= len(text) <= 180 else 0.0
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
        m = re.search(r'(\b\d[\d,\.%kK]*\b.*?)([.!?])', t)
        return (m.group(1).strip() + " — " + t).strip() if m else t

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

    # Try JSON parse first
    try:
        obj = json.loads(response_text)
        if isinstance(obj, dict) and "variants" in obj and isinstance(obj["variants"], list):
            return [str(x).strip() for x in obj["variants"]][:expected]
        if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
            return [x.strip() for x in obj][:expected]
    except Exception:
        pass

    # Fallback: parse numbered lines
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

def _smart_truncate(s: str, limit: int = CHAR_LIMIT) -> str:
    if len(s) <= limit: return s
    cut = s[:limit]
    for p in [r'\.\s', r'!\s', r'\?\s', r'—\s', r',\s', r'\s']:
        m = list(re.finditer(p, cut))
        if m: return cut[:m[-1].end()].strip()
    return cut.strip()

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
    if not cands: return cands
    # Light nudge only (don’t add any generic question text)
    has_stat = any(KEY_NUMBER_RE.search(c) for c in cands)
    out = list(cands)
    # If no stat anywhere, just leave as-is; we rely on LLM + Step B backstops.
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
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    rnd = random.Random(seed)
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
    # 1) ML scores
    with ThreadPoolExecutor(max_workers=min(16, len(variants))) as ex:
        futs = {ex.submit(scorer_ml.score, v): v for v in variants}
        ml_scores = { key: future.result() for key, future in futs.items() }

    base_ml = scorer_ml.score(original)

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
    winner     = blended[0]
    alternates = blended[1:7]

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

def _ml_score_parallel(texts: List[str]) -> Dict[str, SimpleNamespace]:
    # returns {text: Score-like(simple namespace)}
    exec_ = _get_ml_exec()
    results = {}
    for text, lk, rt, rp, comp in exec_.map(_ml_score_worker, texts):
        results[text] = SimpleNamespace(likes=lk, retweets=rt, replies=rp, composite=comp)
    return results

# --- REPLACE: choose_winner with success-criteria aware selection ------------

def choose_winner(original: str, variants: List[Variant], *, scorer: Optional[BaseScorer] = None
                 ) -> Tuple[RankedVariant, List[RankedVariant], Score]:
    """
    Scores original + candidates; only promote a variant if it beats the original.
    Returns (winner, alternates_sorted, original_score).
    """
    scorer = scorer or MLScorer()

    # score original
    base_score = scorer.score(original)

    # ✅ parallel ML scoring without pickling a bound method
    texts = [v.text for v in variants]
    parallel_scores = _ml_score_parallel(texts)  # {text: SimpleNamespace(...)}

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

    return ranked[0], ranked[1:7], base_score

# --- REPLACE: improve_tweet to implement the exact I/O contract --------------

# --- UPDATE improve_tweet signature defaults from config ---
def improve_tweet(
    tweet_text: str,
    mode: Optional[str] = None,
    num_variants: int = NUM_VARIANTS_DEFAULT,
    return_all: bool = False
) -> Dict:
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

    # 4) Generate 5–7 variants with Gemini
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
        response_text = llm.generate_variants(prompt, n=num_variants)  # returns raw text from API
        # Prefer JSON-first parser that accepts a strict JSON object or a fallback numbered list
        if isinstance(response_text, str):
            raw_lines = parse_gemini_variants(response_text, expected=num_variants)
        else:
            # Fallback case where generate_variants already returns a list
            raw_lines = list(response_text)[:num_variants]
    except Exception as e:
        log.warning("LLM generation failed (%s). Falling back to deterministic variants.", e)
        raw_lines = deterministic_variants(tweet_text, n=num_variants)
        fallback_used = True
    timings["llm_ms"] = round((time.time() - t0) * 1000, 2)
    log.debug("Raw variants: %s", raw_lines)

    # --- ADD: if all candidates are too long, run tighten pass to <=260 ---
    if all(len(v.strip()) > CHAR_LIMIT for v in raw_lines):
        log.info("All LLM candidates over 280. Tightening to <=260 and retrying limits.")
        tightened = [tighten_to_limit(v, limit=260, entities=analysis.entities) for v in raw_lines]
        raw_lines = tightened

    # 5) Post-process & enforce constraints
    t0 = time.time()
    orig_entities = analysis.entities  # same object you already have
    processed: List[str] = []
    # --- UPDATE guardrail budgets from config ---
    max_emojis_budget = MAX_EMOJIS_NEWS if analysis.tweet_type in {"news","announcement"} else MAX_EMOJIS_GENERAL
    constraints_applied = [
        f"preserve_hashtags<=orig+{MAX_NEW_HASHTAGS}",
        f"max_emojis<={max_emojis_budget}",
        f"allow_cta={'yes' if ALLOW_CTA_DEFAULT else 'no'}",
        "char_limit<=280", "preserve_links/mentions", "no_new_numbers/dates", "keep_language", "tone_safety"
    ]

    # Track detailed constraint enforcement stats
    constraint_stats = {
        "dropped_extra_hashtags": 0,
        "trimmed_emojis": 0,
        "length_truncated": 0,
        "guardrail_violations": 0
    }

    # Extract key facts once for safety checks
    key_facts = _extract_key_facts(tweet_text)

    for line in raw_lines:
        # sanitize & normalize
        s1 = PostProcessor.sanitize(line, orig_entities, original=tweet_text)
        s1 = PostProcessor.ensure_takeaway_before_link(tweet_text, s1)   # ← NEW (A5)

        # First-pass dedupe
        processed = PostProcessor.dedupe(processed)

        # Ensure length variety (≤60 and 200–240)
        processed = enforce_length_variety(processed, tweet_text, orig_entities)

        # Ensure Step-B roles are covered across the set
        processed = ensure_engagement_mechanics(processed, tweet_text, orig_entities)

        # Optional extra diversity hook (no-op currently)
        processed = ensure_diversity(processed)

        # Trending enrichment (re-enforce limits afterward)
        if TRENDING_ENABLED and processed:
            try:
                trending = get_trending_cached()
                if analysis.tweet_type in ("news", "general") and trending:
                    enriched = maybe_add_trending_hashtag(processed[0], trending)
                    processed[0] = PostProcessor.enforce_limits(
                        enriched, orig_entities,
                        max_new_hashtags=MAX_NEW_HASHTAGS,
                        max_emojis=max_emojis_budget,
                        hard_char_cap=CHAR_LIMIT
                    )
            except Exception as e:
                log.debug("Trending hashtag enrichment failed: %s", e)

        # --- ADD: last length guard ---
        processed = [PostProcessor.smart_truncate(p, limit=CHAR_LIMIT) for p in processed]
        variants = [Variant(text=p) for p in processed]
        timings["post_ms"] = round((time.time() - t0) * 1000, 2)
        try:
            trending_hashtags = get_trending_cached()
            trending_ok = bool(trending_hashtags)
        except Exception:
            trending_ok = False
        
        if analysis.tweet_type in ("news", "general") and trending_ok and processed:
            try:
                processed[0] = maybe_add_trending_hashtag(processed[0], trending_hashtags)
            except Exception as e:
                log.debug("Trending hashtag enrichment failed: %s", e)
    processed = PostProcessor.dedupe(processed)
    # Ensure length variety from Section A (<=60 and 200–240) if the model missed it
    processed = enforce_length_variety(processed, tweet_text, orig_entities)
    log.debug("Processed variants: %s", processed)
    variants = [Variant(text=p) for p in processed]
    timings["post_ms"] = round((time.time() - t0) * 1000, 2)
    
    # --- ADD: sprinkle caches where helpful (example callsites) ---
    # During post-process diversity heuristics:
    base_feat = _light_features(tweet_text)
    cand_feats = {t: _light_features(t) for t in processed}
    # ... prefer at least one variant with a question, one with stat, etc.
    
    processed = ensure_diversity(processed)
    
    # --- USE TRENDING TOGGLE BEFORE OPTIONAL ENRICHMENT ---
    if TRENDING_ENABLED:
        # Optional trending hashtag enrichment
        # Check if trending data is available independently of scoring stack
        trending_ok = False
        try:
            trending_hashtags = get_trending_cached()
            trending_ok = bool(trending_hashtags)
        except Exception:
            trending_ok = False
        
        if analysis.tweet_type in ("news", "general") and trending_ok and processed:
            try:
                processed[0] = maybe_add_trending_hashtag(processed[0], trending_hashtags)
            except Exception as e:
                log.debug("Trending hashtag enrichment failed: %s", e)
    
   
    
    # --- ADD: last length guard ---
    processed = [PostProcessor.smart_truncate(p, limit=CHAR_LIMIT) for p in processed]
    variants = [Variant(text=p) for p in processed]
    timings["post_ms"] = round((time.time() - t0) * 1000, 2)

    # 6) Score & select with mode switch
    # --- UPDATE scoring mode switch & weights ---
    t0 = time.time()
    scorer_ml = MLScorer(alpha=ALPHA_RETWEETS, beta=BETA_REPLIES)

    if SCORING_MODE_DEFAULT == "ml":
        winner, alternates, base_score = choose_winner(tweet_text, variants, scorer=scorer_ml)

    elif SCORING_MODE_DEFAULT == "elo":
        # sample personas for speed using deterministic sampling to avoid bias
        personas_all = load_personas("personas.json")
        personas_sub = _sample_personas(personas_all, tweet_text, k=PERSONA_SAMPLE_SIZE)
        # wrap into Elo path (add original + variants)
        ranked, base_score = rank_with_elo(tweet_text, [v.text for v in variants], personas_sub)
        winner, alternates = ranked[0], ranked[1:7]

    else:  # "blend"
        personas_all = load_personas("personas.json")
        personas_sub = _sample_personas(personas_all, tweet_text, k=PERSONA_SAMPLE_SIZE)
        winner, alternates, base_score = blended_rank(
            original=tweet_text,
            variants=[v.text for v in variants],
            scorer_ml=scorer_ml,
            personas=personas_sub
        )

    timings["score_ms"] = round((time.time() - t0) * 1000, 2)

    # 7) Build deltas & explanations
    # --- Extract results from different scoring modes ---
    if SCORING_MODE_DEFAULT == "ml":
        # winner/alternates already have the right format from choose_winner
        winner_text = winner.variant.text
        winner_score = winner.score
        original_score = base_score
        alternates_list = [
            {"text": alt.variant.text,
             "predicted": {
                 "likes": alt.score.likes,
                 "retweets": alt.score.retweets,
                 "replies": alt.score.replies,
                 "composite": alt.score.composite
             }}
            for alt in alternates
        ] if return_all else []
        personas_for_elo = []  # not used in ML mode
        
    elif SCORING_MODE_DEFAULT == "elo":
        # Extract from ranked results
        winner_text = winner.variant.text
        winner_score = winner.score
        original_score = base_score
        alternates_list = [
            {"text": alt.variant.text,
             "predicted": {
                 "likes": alt.score.likes,
                 "retweets": alt.score.retweets,
                 "replies": alt.score.replies,
                 "composite": alt.score.composite
             }}
            for alt in alternates
        ] if return_all else []
        personas_for_elo = personas_sub
        
    else:  # "blend"
        # Extract from blended results
        winner_text = winner.variant.text
        winner_score = winner.score
        original_score = base_score
        alternates_list = [
            {"text": alt.variant.text,
             "predicted": {
                 "likes": alt.score.likes,
                 "retweets": alt.score.retweets,
                 "replies": alt.score.replies,
                 "composite": alt.score.composite
             }}
            for alt in alternates
        ] if return_all else []
        personas_for_elo = personas_sub

    # --- REPLACE: payload winner/alternates population (variables changed above) --
    likes_delta = (winner_score.likes - original_score.likes)
    comp_delta  = (winner_score.composite - original_score.composite)
    pct_improve = 0.0 if original_score.composite == 0 else (comp_delta / original_score.composite)

    # --- ADD: no-uplift notice & guardrail logging ---
    notices = []
    if winner_text == tweet_text or winner_score.composite <= original_score.composite:
        notices.append("No predicted improvement over original; showing strongest candidate.")

    if fallback_used:
        notices.append("LLM unavailable; used deterministic fallback variants.")

    why = (f"Highest {SCORING_MODE_DEFAULT} score (ML: composite | Elo: persona rank | Blend: 70% ML + 30% Elo) "
           "AND higher engagement than original."
           if winner_text != tweet_text
           else "No generated version beat the original on metrics; keeping the original.")

    explanations = {
        "why_this_won": why,
        "detected_type": analysis.tweet_type,
        "applied_transformations": [
            "kept links/@mentions/numbers/dates",
            "tightened to ≤280 chars",
            "capped emojis/hashtags",
            "prompted for clearer hook and benefit",
            "diversified tones/forms (question/imperative/stat)"
        ],
        "constraints_applied": constraints_applied,
        "constraint_stats": constraint_stats,  # detailed enforcement statistics
        "fallback_used": fallback_used,  # track if deterministic variants were used
        "scoring": {
            "mode": SCORING_MODE_DEFAULT,
            "weights": {"ml": BLEND_W_ML, "elo": BLEND_W_ELO} if SCORING_MODE_DEFAULT == "blend" else {},
            "ml_formula": f"likes + {ALPHA_RETWEETS}·retweets + {BETA_REPLIES}·replies",
            "elo_personas_used": min(PERSONA_SAMPLE_SIZE, len(personas_for_elo)) if personas_for_elo else 0
        }
    }

    guardrails = [
        "Preserved links and @mentions; original hashtags retained when present; added ≤1 new hashtag.",
        "Hard cap at 280 characters (Unicode).",
        "Emojis capped at 2 (1 for news).",
        "Did not alter numbers or dates; no new factual claims.",
        "Kept original language/script; did not translate.",
        "Kept tone respectful; avoided medical/financial advice changes."
    ]

    timings["total_ms"] = round((time.time() - t_total) * 1000, 2)

    # 8) Return EXACT contract for your frontend
    payload = {
        "winner": {
            "text": winner_text,
            "predicted": {
                "likes": winner_score.likes,
                "retweets": winner_score.retweets,
                "replies": winner_score.replies,
                "composite": winner_score.composite
            },
            "delta_vs_original": {
                "likes_delta": likes_delta,
                # pct is composite % improvement for stability across metrics
                "pct": f"{pct_improve * 100:.1f}%"
            }
        },
        "alternates": [] if not return_all else alternates_list,
        "explanations": explanations,
        "guardrails": guardrails,
        "notices": notices,
        "timings": timings
    }
    
    # --- ADD: telemetry about fallbacks ---
    payload.setdefault("meta", {})
    payload["meta"].update({
        "fallback_used": fallback_used,
        "num_candidates": len(variants)
    })
    
    # Log telemetry for offline evaluation  
    selection_mode = SCORING_MODE_DEFAULT
    winner_for_log = RankedVariant(variant=Variant(text=winner_text), score=winner_score)
    _log_telemetry(tweet_text, winner_for_log, timings, analysis, original_score, selection_mode)
    
    return payload

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
    parser.add_argument("--num_variants", type=int, default=6, help="Number of variants")
    args = parser.parse_args()
    
    result = improve_tweet(
        tweet_text=args.tweet_text,
        mode=args.mode,
        num_variants=args.num_variants,
        return_all=args.return_all,
    )
    
    print("\n=== WINNER ===")
    print(result["winner"]["text"])
    print("Predicted:", result["winner"]["predicted"])
    print("Delta vs original:", result["winner"]["delta_vs_original"])
    
    print("\n=== ALTERNATES ===")
    for alt in result["alternates"]:
        print("-", alt["text"], alt["predicted"])
    
    print("\n=== CONTEXT ===")
    print("Why this won:", result["explanations"]["why_this_won"])
    print("Applied transforms:", result["explanations"]["applied_transformations"])
    print("Guardrails:", result["guardrails"])
