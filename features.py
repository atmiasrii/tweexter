import re
import string
import datetime
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer

# Import new modules for optional extended features
try:
    from topic_category_hf import get_topic_category
    from trending_hashtags import count_trending_hashtags
except ImportError:
    # Fallback functions if modules not available
    def get_topic_category(text, threshold=0.3):
        return 0
    def count_trending_hashtags(text):
        return 0

sia = SentimentIntensityAnalyzer()

# Optional: Hugging Face (free, local)
HF_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_tokenizer = None
_hf_model = None

def _hf_preprocess(text: str) -> str:
    toks = []
    for t in text.split():
        if t.startswith("@") and len(t) > 1: t = "@user"
        if t.startswith("http"): t = "http"
        toks.append(t)
    return " ".join(toks)

def _lazy_load_hf():
    global _tokenizer, _hf_model
    if _tokenizer is None or _hf_model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
        _tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
        _hf_model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_ID)
        _hf_model.eval()

def _sentiment_compound(text: str) -> float:
    """Prefer Twitter-RoBERTa; fallback to VADER if unavailable."""
    try:
        _lazy_load_hf()
        from scipy.special import softmax
        import torch
        txt = _hf_preprocess(text)
        inputs = _tokenizer(txt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = _hf_model(**inputs).logits[0].cpu().numpy()
        p_neg, p_neu, p_pos = softmax(logits).tolist()
        # map to [-1, 1]-ish scale to replace VADER's 'compound'
        return float(p_pos - p_neg)
    except Exception:
        # fallback: VADER
        return float(sia.polarity_scores(text)["compound"])

MODEL_FEATURES = [
    'char_count', 'word_count', 'sentence_count', 'avg_word_length', 'uppercase_ratio',
    'hashtag_count', 'mention_count', 'exclamation_count', 'question_count',
    'sentiment_compound', 'hour', 'day_of_week', 'is_weekend', 'sentiment_length',
    'hashtag_word_ratio', 'hour_sin', 'hour_cos', 'char_count_squared', 'sentiment_word_interaction',
    'is_prime_hour', 'mention_ratio', 'is_short_viral', 'viral_potential', 'is_extreme_viral',
    'viral_interaction', 'viral_sentiment', 'viral_length'
]
# 27 standardized features (removed url_present, emoji_count, has_emoji, has_url)

def extract_features_from_text(text, tweet_time=None):
    # --- Tokenization ---
    words = re.findall(r'\b\w+\b', text)
    word_count = len(words)
    char_count = len(text)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = max(1, len([s for s in sentences if s.strip()]))
    avg_word_length = np.mean([len(w) for w in words]) if word_count else 0

    # --- Casing & Punctuation ---
    uppercase_chars = sum(1 for c in text if c.isupper())
    uppercase_ratio = uppercase_chars / char_count if char_count else 0
    exclamation_count = text.count('!')
    question_count = text.count('?')

    # --- Hashtags & Mentions ---
    hashtag_count = len(re.findall(r'(?<!\w)#[\w_]+', text))
    mention_count = len(re.findall(r'(?<!\w)@[\w_]+', text))

    # --- Special Features --- (removed url_present, emoji_count, has_emoji, has_url)

    # --- Sentiment ---
    sentiment_compound = _sentiment_compound(text)

    # --- Time Features (assume now if not provided) ---
    if tweet_time is None:
        now = datetime.datetime.now()
    else:
        now = tweet_time
    hour = now.hour
    day_of_week = now.weekday()
    is_weekend = int(day_of_week in [5, 6])

    # --- Derived Features ---
    sentiment_length = sentiment_compound * char_count
    hashtag_word_ratio = hashtag_count / (word_count + 1)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    char_count_squared = char_count ** 2
    sentiment_word_interaction = sentiment_compound * word_count
    is_prime_hour = int(hour in [7, 8, 9, 10, 18, 19, 20, 21])
    mention_ratio = mention_count / (word_count + 1)

    # --- Viral Heuristics ---
    is_short_viral = int(
        char_count < 100 and 
        sentiment_compound > 0.3 and 
        hashtag_count == 0 and 
        mention_count == 0
    )
    viral_potential = int(
        char_count < 100 and 
        sentiment_compound > 0.3 and 
        hashtag_count == 0
    )

    is_extreme_viral = 0  # Placeholder for scoring, updated elsewhere if needed
    viral_interaction = char_count * is_extreme_viral
    viral_sentiment = sentiment_compound * is_extreme_viral
    viral_length = char_count * is_extreme_viral

    # --- Create feature dict in model order ---
    features = {
        "char_count": char_count,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "uppercase_ratio": uppercase_ratio,
        "hashtag_count": hashtag_count,
        "mention_count": mention_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "sentiment_compound": sentiment_compound,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "sentiment_length": sentiment_length,
        "hashtag_word_ratio": hashtag_word_ratio,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "char_count_squared": char_count_squared,
        "sentiment_word_interaction": sentiment_word_interaction,
        "is_prime_hour": is_prime_hour,
        "mention_ratio": mention_ratio,
        "is_short_viral": is_short_viral,
        "viral_potential": viral_potential,
        "is_extreme_viral": is_extreme_viral,
        "viral_interaction": viral_interaction,
        "viral_sentiment": viral_sentiment,
        "viral_length": viral_length,
    }
    # Return only the features in the exact required order!
    ordered_features = {k: features[k] for k in MODEL_FEATURES}
    return ordered_features

def extract_extended_features_from_text(text, tweet_time=None):
    """
    Extract features including topic_category and trending_hashtag_count for NEW models.
    Falls back gracefully if new modules are not available.
    """
    # Get base features
    features = extract_features_from_text(text, tweet_time)
    
    # Add extended features
    try:
        features['topic_category'] = get_topic_category(text)
        features['trending_hashtag_count'] = count_trending_hashtags(text)
    except Exception:
        features['topic_category'] = 0
        features['trending_hashtag_count'] = 0
    
    # Add other features NEW models expect
    features['text_length'] = len(text)
    features['has_media'] = 0  # placeholder - update upstream if media info available
    features['has_url'] = 1 if ('http://' in text.lower() or 'https://' in text.lower()) else 0
    features['has_news_url'] = 0  # placeholder
    
    # Reply-specific features
    text_lower = text.lower()
    features['question_word_count'] = sum(1 for w in ["what","how","why","when","where","who","which"] if w in text_lower)
    features['cta_word_count'] = sum(1 for w in ["please","help","share","comment","reply","thoughts"] if w in text_lower)
    
    # Emotion features (placeholder)
    for emo in ["anger","fear","joy","sadness","surprise"]:
        features[f'emotion_{emo}'] = 0.0
    
    features['sentiment_subjectivity'] = 0.5  # placeholder
    
    return features

# --- Legacy support for backward compatibility ---
# Features that calibrator expects (must match calibrator_candidates)
VIRAL_FEATURES = [
    'question_count', 'exclamation_count', 'sentiment_compound',
    'char_count', 'hashtag_count', 'mention_count'
]
# These features are used by the calibrator and all exist in MODEL_FEATURES

def select_features(row):
    """Legacy function for viral feature selection"""
    return {k: row[k] for k in VIRAL_FEATURES if k in row}

# USAGE EXAMPLE:
# feats = extract_features_from_text("Sample tweet here!")
# print(feats)
