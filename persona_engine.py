# Add parallel processing
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from datetime import datetime
from collections import defaultdict
import numpy as np
import json
import re
import random
import os
import math
from fuzzywuzzy import fuzz
# Add lru_cache for persona cache


# RapidFuzz (drop fuzzywuzzy)
try:
    from rapidfuzz.fuzz import partial_ratio as _partial_ratio
except Exception:
    # Cheap fallback (install rapidfuzz for real speed)
    def _partial_ratio(a, b):
        return 100 if (a in b or b in a) else 0

# === Fixed, reusable pool for persona simulation ===
# Use processes + sane pool + chunking (replace 5000 threads)
USE_PROCESSES = os.getenv("PE_USE_PROCESSES", "1") == "1"
if USE_PROCESSES:
    from concurrent.futures import ProcessPoolExecutor as Executor
    name_prefix = None  # processes don't use thread names
else:
    from concurrent.futures import ThreadPoolExecutor as Executor
    name_prefix = "persona"

POOL_SIZE = min(32, (os.cpu_count() or 4) * (2 if USE_PROCESSES else 8))

# Fix: ProcessPoolExecutor doesn't accept thread_name_prefix
if USE_PROCESSES:
    EXECUTOR = Executor(max_workers=POOL_SIZE)
else:
    EXECUTOR = Executor(max_workers=POOL_SIZE, thread_name_prefix=name_prefix)

PE_DEBUG = os.getenv("PE_DEBUG", "0") == "1"
def _log(*a, **k):
    if PE_DEBUG: print(*a, **k)

_log(f"✅ PersonaEngine: created {'process' if USE_PROCESSES else 'thread'} pool with max_workers={POOL_SIZE}")

# === Performance flags & lazy NLP loaders ===
from functools import lru_cache
import itertools
_SPLIT_RE = re.compile(r'[.,!?;:\n]+')
_QWORDS = ('?', 'why', 'how', 'what', 'thoughts', 'opinion')

# Batch personas per worker (big cut in scheduling/pickling)
def _process_chunk(chunk, tweet, tweet_lower, words, phrases, all_content, sentiment):
    return [
        simulate_persona_engagement(
            tweet, p,
            tweet_lower=tweet_lower,
            words=words,
            phrases=phrases,
            all_content=all_content,
            sentiment=sentiment
        ) for p in chunk
    ]

USE_TEXTBLOB = os.getenv("USE_TEXTBLOB", "0") == "1"     # turn on TextBlob via env if you want
SPACY_ENABLED = os.getenv("SPACY_ENABLED", "0") == "1"   # only load spaCy if explicitly enabled
SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")

@lru_cache(maxsize=1)
def get_nlp():
    """Lazy-load spaCy the first time it's actually needed."""
    if not SPACY_ENABLED:
        return None
    try:
        import spacy
        return spacy.load(SPACY_MODEL)
    except Exception as e:
        print(f"⚠️ spaCy load skipped/failing ({e}); falling back to basic matching.")
        return None

@lru_cache(maxsize=1)
def get_textblob():
    """Lazy import TextBlob only if explicitly requested."""
    if not USE_TEXTBLOB:
        return None
    try:
        from textblob import TextBlob
        return TextBlob
    except Exception as e:
        print(f"⚠️ TextBlob not available ({e}); using lightweight sentiment.")
        return None

POS_WORDS = {
    "great","good","love","amazing","awesome","fantastic","win","profit","success","well","happy","excited","🚀","🔥","💯"
}
NEG_WORDS = {
    "bad","hate","terrible","awful","fail","loss","angry","sad","worse","problem","issue","😞","💀"
}

def quick_sentiment(text: str) -> float:
    """
    Very light sentiment fallback: (#pos - #neg) / (len(tokens)+3) -> [-1, +1]ish
    If USE_TEXTBLOB=1 and TextBlob is present, use it instead.
    """
    tb = get_textblob()
    if tb is not None:
        try:
            return float(tb(text).sentiment.polarity)
        except Exception:
            pass
    t = (text or "").lower().split()
    if not t:
        return 0.0
    pos = sum(1 for w in t if w in POS_WORDS)
    neg = sum(1 for w in t if w in NEG_WORDS)
    return max(-1.0, min(1.0, (pos - neg) / (len(t) + 3)))

# Import new trending hashtags module
try:
    from trending_hashtags import get_trending_hashtags
    TRENDING_MODULE_AVAILABLE = True
except ImportError:
    TRENDING_MODULE_AVAILABLE = False
    _log("⚠️ trending_hashtags module not available - using manual hashtag setting")

class FeedbackLearner:
    def __init__(self):
        self.history = defaultdict(list)
        
    def update(self, persona_id, predicted, actual):
        """Update persona-specific feedback"""
        if actual > 0:  # Avoid division by zero
            error = (predicted - actual) / actual
            self.history[persona_id].append(error)
            # Keep only last 50 records per persona
            self.history[persona_id] = self.history[persona_id][-50:]
        
    def get_adjustment(self, persona_id):
        """Get error correction factor for persona"""
        if persona_id not in self.history or not self.history[persona_id]:
            return 1.0
        avg_error = sum(self.history[persona_id]) / len(self.history[persona_id])
        return max(0.5, min(1.5, 1 - (avg_error * 0.3)))  # Partial correction with bounds

class EnhancedPersonaEngine:
    def __init__(self, ml_models=None):
        # Initialize with ML models for NLP features
        self.ml_models = ml_models or {}
        self.user_engagement_history = defaultdict(list)
        self.topic_frequency = defaultdict(int)
        self.feedback_learner = FeedbackLearner()
        
        # Calibration data for realistic scaling
        self.calibration_data = {
            'avg_actual_likes': 50,  # Average likes from your dataset
            'avg_actual_retweets': 15,  # Average retweets from your dataset
            'avg_actual_replies': 8,   # Average replies from your dataset
            'baseline_reach': 1000     # Average reach per tweet
        }
        
        # Viral content thresholds
        self.viral_triggers = {
            'trending_hashtags': set(),
            'news_domains': {'bbc.com', 'cnn.com', 'nytimes.com', 'reuters.com', 'techcrunch.com'}
        }
        
        # User tiers configuration with realistic base rates
        self.user_tiers = {
            'casual': {'base': 0.7, 'decay': 0.9},
            'regular': {'base': 1.0, 'decay': 1.0},
            'influencer': {'base': 1.3, 'decay': 1.1}
        }

    def enhanced_content_match(self, tweet, persona):
        """Advanced content matching using NLP and fuzzy matching, if spaCy is enabled."""
        nlp = get_nlp()
        if nlp is not None:
            return self._nlp_content_match(tweet, persona, nlp)
        else:
            return self._basic_content_match(tweet, persona)
    
    def _nlp_content_match(self, tweet, persona, nlp):
        """NLP-based content matching when spaCy is available (lazy-loaded)."""
        try:
            doc = nlp(tweet)
            # 1) Topic matching (if you have ML model)
            topics = set(self.ml_models['topic'].predict(tweet)) if 'topic' in self.ml_models else set()
            topic_score = len(topics & persona.content_types) * 0.3 if topics else 0.0
            # 2) Keyword expansion with fuzzy matching (keep your existing logic)
            interests_matched = 0
            for token in doc:
                for interest in persona.interests:
                    try:
                        if _partial_ratio(interest, token.text.lower()) > 75:
                            interests_matched += 1
                            break
                    except Exception:
                        if interest in token.text.lower():
                            interests_matched += 1
                            break
            # 3) Semantic-ish triggers with token similarity
            triggers_matched = 0
            for token in doc:
                for trigger in persona.triggers:
                    try:
                        trig = nlp(trigger)
                        if trig and token.has_vector and trig[0].has_vector and token.similarity(trig[0]) > 0.65:
                            triggers_matched += 1
                            break
                    except Exception:
                        if trigger in token.text.lower():
                            triggers_matched += 1
                            break
            final_score = min(1.0, topic_score + interests_matched * 0.2 + triggers_matched * 0.5)
            return final_score
        except Exception as e:
            print(f"NLP matching failed: {e}. Using basic matching.")
            return self._basic_content_match(tweet, persona)
    
    def _basic_content_match(self, tweet, persona):
        tweet_lower = tweet.lower()

        # Simple keyword matching (fields already normalized)
        interest_matches = sum(1 for interest in persona.interests if interest in tweet_lower)
        trigger_matches = sum(1 for trigger in persona.triggers if trigger in tweet_lower)
        content_matches = sum(1 for content in persona.content_types if content in tweet_lower)

        # Normalize scores
        max_possible = max(1, len(persona.interests) + len(persona.triggers) + len(persona.content_types))
        total_matches = interest_matches + trigger_matches + content_matches

        return min(1.0, total_matches / max_possible)

    def get_base_rates(self, persona):
        """Dynamic base rates based on persona engagement style"""
        base_rates = {
            'lurker': {'like': 0.01, 'retweet': 0.002, 'reply': 0.001},
            'casual': {'like': 0.02, 'retweet': 0.005, 'reply': 0.003},
            'average': {'like': 0.03, 'retweet': 0.01, 'reply': 0.005},
            'proactive': {'like': 0.05, 'retweet': 0.02, 'reply': 0.01},
            'influencer': {'like': 0.08, 'retweet': 0.03, 'reply': 0.015}
        }
        
        style = getattr(persona, 'engagement_style', 'average')
        return base_rates.get(style, base_rates['average'])

    def calculate_scale_factor(self):
        """Calibrate using actual dataset statistics"""
        # This should be updated with your real data
        cal = self.calibration_data
        
        # Estimate based on typical engagement rates
        estimated_likes_per_1000 = 30  # 3% engagement rate
        estimated_rts_per_1000 = 10    # 1% retweet rate
        estimated_replies_per_1000 = 5  # 0.5% reply rate
        
        return {
            'likes': cal['avg_actual_likes'] / estimated_likes_per_1000,
            'retweets': cal['avg_actual_retweets'] / estimated_rts_per_1000,
            'replies': cal['avg_actual_replies'] / estimated_replies_per_1000
        }

    def predict_virality(self, tweet_text, user_id=None, user_tier='regular', debug=False):
        """Main prediction method with all enhancements"""
        if debug:
            print(f"\n=== Debug: Predicting for user {user_id} ({user_tier}) ===")
            print(f"Tweet: {tweet_text[:100]}...")
        
        # Content analysis
        content_features = self._analyze_content(tweet_text)
        user_profile = self._get_user_profile(user_id, user_tier)
        
        if debug:
            print(f"Content features: {content_features}")
            print(f"User profile: {user_profile}")
        
        # Base prediction using realistic rates
        base_pred = self._calculate_base_prediction(content_features, user_profile)
        
        # Apply viral boosting
        viral_boost = self._calculate_viral_boost(content_features)
        
        # Apply decay for overused topics
        decay_factor = self._calculate_decay(content_features['topics'])
        
        # Apply feedback learning adjustment
        feedback_adjustment = self.feedback_learner.get_adjustment(user_id) if user_id else 1.0
        
        # Final prediction with realistic scaling
        scale_factors = self.calculate_scale_factor()
        
        adjusted_pred = {
            'likes': base_pred['likes'] * viral_boost * decay_factor * feedback_adjustment * scale_factors['likes'],
            'retweets': base_pred['retweets'] * viral_boost * decay_factor * feedback_adjustment * scale_factors['retweets'],
            'replies': base_pred['replies'] * content_features['reply_boost'] * decay_factor * feedback_adjustment * scale_factors['replies']
        }
        
        # Dynamic scaling based on historical accuracy
        final_pred = self._apply_dynamic_scaling(adjusted_pred, user_id)
        
        if debug:
            print(f"Base prediction: {base_pred}")
            print(f"Viral boost: {viral_boost:.2f}")
            print(f"Decay factor: {decay_factor:.2f}")
            print(f"Feedback adjustment: {feedback_adjustment:.2f}")
            print(f"Final prediction: {final_pred}")
        
        return {
            'predictions': final_pred,
            'content_features': content_features,
            'viral_score': viral_boost,
            'decay_factor': decay_factor,
            'feedback_adjustment': feedback_adjustment
        }

    def _analyze_content(self, text):
        """Extract all NLP features from tweet text"""
        # Sentiment analysis
        sentiment = quick_sentiment(text)
        
        # Topic classification (using your ML model)
        topics = self.ml_models['topic'].predict(text) if 'topic' in self.ml_models else self._basic_topic_detection(text)
        
        # Viral triggers
        hashtags = set(re.findall(r'#(\w+)', text.lower()))
        trending_match = len(hashtags & self.viral_triggers['trending_hashtags']) / max(1, len(hashtags)) if hashtags else 0
        
        # News detection
        urls = re.findall(r'https?://[^\s]+', text)
        news_urls = any(any(nd in url for nd in self.viral_triggers['news_domains']) for url in urls)
        
        # Reply triggers
        question_words = any(q in text.lower() for q in ['?', 'why', 'how', 'what', 'thoughts', 'opinion'])
        cta_phrases = any(p in text.lower() for p in ['thoughts?', 'agree?', 'opinions?', 'what do you think'])
        
        return {
            'sentiment': sentiment,
            'topics': topics,
            'trending_match': trending_match,
            'has_news': news_urls,
            'question': question_words,
            'cta': cta_phrases,
            'reply_boost': 1.4 if question_words else (1.2 if cta_phrases else 1.0),
            'hashtag_count': len(hashtags),
            'text_length': len(text)
        }

    def _basic_topic_detection(self, text):
        """Basic topic detection when ML model isn't available"""
        topics = []
        business_words = ['business', 'money', 'invest', 'profit', 'entrepreneur', 'startup']
        tech_words = ['tech', 'ai', 'software', 'coding', 'digital', 'innovation']
        lifestyle_words = ['health', 'fitness', 'travel', 'food', 'lifestyle']
        
        text_lower = text.lower()
        if any(word in text_lower for word in business_words):
            topics.append('business')
        if any(word in text_lower for word in tech_words):
            topics.append('tech')
        if any(word in text_lower for word in lifestyle_words):
            topics.append('lifestyle')
        
        return topics or ['general']

    def _get_user_profile(self, user_id, tier):
        """Get user-specific parameters"""
        # Get historical accuracy for this user
        hist_errors = self.user_engagement_history.get(user_id, [0.2])
        hist_error = sum(hist_errors) / len(hist_errors) if hist_errors else 0.2
        
        return {
            'base_multiplier': self.user_tiers[tier]['base'],
            'decay_rate': self.user_tiers[tier]['decay'],
            'error_adjustment': 0.9 if hist_error > 0.3 else 1.1 if hist_error < 0.1 else 1.0
        }

    def _calculate_base_prediction(self, features, user):
        """Content-aware base prediction with realistic rates"""
        # Get dynamic base rates based on user tier
        base_rates = {
            'casual': {'like': 0.015, 'retweet': 0.003, 'reply': 0.002},
            'regular': {'like': 0.025, 'retweet': 0.008, 'reply': 0.004},
            'influencer': {'like': 0.040, 'retweet': 0.015, 'reply': 0.008}
        }
        
        user_tier = 'regular'  # Default
        if user['base_multiplier'] < 0.8:
            user_tier = 'casual'
        elif user['base_multiplier'] > 1.2:
            user_tier = 'influencer'
            
        base_rate = base_rates[user_tier]
        
        # Content quality multiplier (0.5 to 2.0 range)
        content_quality = (
            0.3 * max(0, features['sentiment']) + 
            0.4 * min(1.0, len(features['topics']) * 0.3) + 
            0.3 * features['trending_match']
        )
        content_multiplier = 0.5 + (1.5 * content_quality)  # Range: 0.5 to 2.0
        
        # Text length factor (optimal around 100-150 chars)
        length_factor = 1.0
        if 80 <= features['text_length'] <= 180:
            length_factor = 1.2
        elif features['text_length'] > 280:
            length_factor = 0.8
        elif features['text_length'] < 30:
            length_factor = 0.7
        
        return {
            'likes': base_rate['like'] * content_multiplier * length_factor * user['base_multiplier'],
            'retweets': base_rate['retweet'] * content_multiplier * length_factor * user['base_multiplier'],
            'replies': base_rate['reply'] * content_multiplier * length_factor * user['base_multiplier']
        }

    def _calculate_viral_boost(self, features):
        """Calculate viral potential score (0.5-3.0 range)"""
        score = 1.0
        
        # News boost
        if features['has_news']:
            score *= 1.5
            
        # Trending hashtag boost
        if features['trending_match'] > 0:
            score *= 1 + (features['trending_match'] * 0.8)
            
        # Emotion boost
        if features['sentiment'] > 0.4:  # Very positive
            score *= 1.3
        elif features['sentiment'] > 0.1:  # Positive
            score *= 1.1
        elif features['sentiment'] < -0.4:  # Very controversial
            score *= 1.4
        elif features['sentiment'] < -0.1:  # Controversial
            score *= 1.2
            
        # Question/engagement boost
        if features['question']:
            score *= 1.3
        elif features['cta']:
            score *= 1.2
            
        return max(0.5, min(score, 3.0))  # Constrain to reasonable range

    def _calculate_decay(self, topics):
        """Reduce predictions for overused topics"""
        decay = 1.0
        for topic in topics:
            self.topic_frequency[topic] += 1
            if self.topic_frequency[topic] > 10:  # If topic used >10 times today
                decay *= 0.95  # 5% reduction per overuse
        return max(decay, 0.7)  # Minimum 70% of original

    def _apply_dynamic_scaling(self, pred, user_id):
        """Adjust based on historical accuracy"""
        if not user_id:
            return pred
            
        # Get this user's average error
        hist_errors = self.user_engagement_history.get(user_id, [0.2])
        avg_error = sum(hist_errors) / len(hist_errors) if hist_errors else 0.2
        
        # Apply error correction
        adjustment = 0.85 if avg_error > 0.4 else 1.15 if avg_error < 0.1 else 1.0
        
        return {
            'likes': pred['likes'] * adjustment,
            'retweets': pred['retweets'] * adjustment,
            'replies': pred['replies'] * adjustment
        }

    def update_feedback(self, user_id, actual_engagements, predicted):
        """Update user model with actual results using FeedbackLearner"""
        # Update feedback for each engagement type
        if 'likes' in actual_engagements and actual_engagements['likes'] > 0:
            self.feedback_learner.update(f"{user_id}_likes", predicted['likes'], actual_engagements['likes'])
        
        if 'retweets' in actual_engagements and actual_engagements['retweets'] > 0:
            self.feedback_learner.update(f"{user_id}_retweets", predicted['retweets'], actual_engagements['retweets'])
            
        if 'replies' in actual_engagements and actual_engagements['replies'] > 0:
            self.feedback_learner.update(f"{user_id}_replies", predicted.get('replies', 0), actual_engagements['replies'])
        
        # Also update the legacy system for backward compatibility
        like_error = abs(actual_engagements.get('likes', 0) - predicted['likes']) / max(1, actual_engagements.get('likes', 1))
        rt_error = abs(actual_engagements.get('retweets', 0) - predicted['retweets']) / max(1, actual_engagements.get('retweets', 1))
        reply_error = abs(actual_engagements.get('replies', 0) - predicted.get('replies', 0)) / max(1, actual_engagements.get('replies', 1))
        
        avg_error = (like_error + rt_error + reply_error) / 3
        self.user_engagement_history[user_id].append(avg_error)
        self.user_engagement_history[user_id] = self.user_engagement_history[user_id][-20:]

    def set_trending_hashtags(self, hashtags=None):
        """Update trending hashtags (call this periodically with real data)"""
        if hashtags is not None:
            # Use manually provided hashtags
            self.viral_triggers['trending_hashtags'] = set(tag.lower() for tag in hashtags)
        elif TRENDING_MODULE_AVAILABLE:
            # Use automatic trending hashtags from module
            try:
                trending_set = get_trending_hashtags()
                self.viral_triggers['trending_hashtags'] = trending_set
                print(f"✅ Auto-updated with {len(trending_set)} trending hashtags")
            except Exception as e:
                print(f"⚠️ Failed to auto-update trending hashtags: {e}")
        else:
            _log("⚠️ No trending hashtags available - provide hashtags manually or install trending_hashtags module")

    def reset_daily_counters(self):
        """Reset topic frequency counters (call this daily)"""
        self.topic_frequency.clear()

def aggregate(results, personas=None, demographic_weights=None):
    total_likes = 0
    total_rts = 0
    total_replies = 0
    engaged_personas = []
    for i, engagement in enumerate(results):
        if personas:
            persona = personas[i]
            weight = demographic_weights.get(persona.id, 1.0) if demographic_weights else 1.0
        else:
            weight = 1.0
            persona = None
        total_likes += engagement['like'] * weight
        total_rts += engagement['retweet'] * weight
        total_replies += engagement['reply'] * weight
        if engagement['like'] > 0.02 or engagement['retweet'] > 0.008 or engagement['reply'] > 0.005:
            if persona:
                engaged_personas.append({
                    'id': persona.id,
                    'demographic': getattr(persona, 'demographic', {}),
                    'engagement': engagement
                })
    return {
        'persona_likes': total_likes * 10,
        'persona_rts': total_rts * 10,
        'persona_replies': total_replies * 10,
        'engaged_personas': engaged_personas[:20]
    }

def parallel_engagement(tweet, personas, demographic_weights=None):
    # Precompute shared tweet features once
    tweet_lower = tweet.lower()
    words = {w.strip('.,!?":;').lower() for w in tweet.split()}
    phrases = {p.strip().lower() for p in _SPLIT_RE.split(tweet) if p.strip()}
    all_content = words | phrases
    sentiment = quick_sentiment(tweet)

    try:
        _log(f"✅ PersonaEngine: using {len(personas)} personas (pool={POOL_SIZE})")
    except Exception:
        pass

    CHUNK = 512
    chunks = [personas[i:i+CHUNK] for i in range(0, len(personas), CHUNK)]

    mapped = EXECUTOR.map(
        _process_chunk,
        chunks,
        itertools.repeat(tweet),
        itertools.repeat(tweet_lower),
        itertools.repeat(words),
        itertools.repeat(phrases),
        itertools.repeat(all_content),
        itertools.repeat(sentiment),
    )
    results = [eng for sub in mapped for eng in sub]
    return aggregate(results, personas, demographic_weights)


import json
import re
import random
from collections import namedtuple
import math

# Enhanced persona structure with emotional analysis
Persona = namedtuple('Persona', [
    'id', 'interests', 'triggers', 'pet_peeves', 'content_types',
    'emotional_triggers', 'dealbreakers', 'demographic', 'follower_weight',
    'base_engagement', 'engagement_style', 'linguistic_style'
])

# ELO Constants
BASE_ELO = 1400
ELO_K = 32  # Controls rating update sensitivity

def _load_personas_uncached(json_path):
    # Try to create sample personas if file doesn't exist
    create_sample_personas_if_missing(json_path)
    personas = []
    try:
        with open(json_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        _log(f"Warning: {json_path} not found. Creating minimal personas for testing.")
        data = create_sample_personas_if_missing(json_path)
    except json.JSONDecodeError:
        _log(f"Warning: {json_path} contains invalid JSON. Using minimal personas.")
        data = []
    if not data:
        _log("Warning: No persona data available. Using default personas.")
        return []
    for p in data:
        # Enhanced follower impact calculation
        follower_str = p.get('behavior', {}).get('follower_count', '5k-5k')
        if 'k' in follower_str:
            nums = [float(x.replace('k', '')) * 1000 for x in re.findall(r'[\d.]+k', follower_str)]
            if len(nums) == 2:
                low, high = nums
                follower_mid = (low + high) / 2
            elif len(nums) == 1:
                follower_mid = nums[0]
            else:
                follower_mid = 5000
        else:
            nums = [int(x) for x in re.findall(r'\d+', follower_str)]
            follower_mid = sum(nums)/len(nums) if nums else 5000
        follower_weight = max(0.5, min(2.0, 1 + (follower_mid - 5000) / 10000))
        psych = p.get('psychographics', {})
        behavior = p.get('behavior', {})
        engagement = p.get('engagement', {})
        linguistic = p.get('linguistic', {})
        content_types = frozenset(s.lower() for s in behavior.get('content_type', []))
        content_triggers = set(engagement.get('content_triggers', []))
        emotional_triggers = frozenset(s.lower() for s in engagement.get('emotional_triggers', []))
        dealbreakers = frozenset(s.lower() for s in engagement.get('dealbreakers', []))
        pet_peeves = frozenset(s.lower() for s in behavior.get('pet_peeves', []))
        personas.append(Persona(
            id=p.get('id', len(personas)),
            interests=frozenset(s.lower() for s in psych.get('interests', [])),
            triggers=frozenset(s.lower() for s in (content_triggers | set(psych.get('triggers', [])))),
            pet_peeves=pet_peeves,
            content_types=content_types,
            emotional_triggers=emotional_triggers,
            dealbreakers=dealbreakers,
            demographic=p.get('demographics', {}),
            follower_weight=follower_weight,
            base_engagement={
                'like': 0.03,
                'retweet': 0.01
            },
            engagement_style=engagement.get('style', 'neutral'),
            linguistic_style={
                'emoji_freq': linguistic.get('emoji_freq', 3),
                'tone': linguistic.get('tone', 'neutral')
            }
        ))
    _log(f"✅ PersonaEngine: loaded {len(personas)} personas from {json_path}")
    return personas

# --- Cached, mtime-aware loader wrapper ---
def load_personas(json_path="personas.json"):
    """
    Cached facade. We key the cache by (path, mtime) so whenever
    the file changes on disk, the cache is naturally invalidated.
    """
    try:
        mtime = os.path.getmtime(json_path)
    except OSError:
        # File missing or unreadable -> use -1 so it still keys the cache
        mtime = -1
    return _load_personas_cached(json_path, mtime)


@lru_cache(maxsize=4)
def _load_personas_cached(json_path: str, mtime: float):
    # IMPORTANT: Do not use 'mtime' inside except for cache keying.
    # Any body change here will still reuse the original heavy builder.
    _log(f"✅ Persona cache: (re)loading from {json_path} (mtime={mtime})")
    return _load_personas_uncached(json_path)


def invalidate_persona_cache():
    _load_personas_cached.cache_clear()
    _log("🧹 Persona cache cleared")


def analyze_emotion(tweet_text):
    """Enhanced emotion analysis using quick_sentiment"""
    polarity = quick_sentiment(tweet_text)
    if polarity > 0.3:
        return "positive"
    elif polarity < -0.3:
        return "negative"
    else:
        return "neutral"

def match_style(tweet_text, persona):
    """Check if tweet matches persona's linguistic style"""
    emoji_count = sum(1 for c in tweet_text if c in "😀😂❤️🔥👍👎")
    expected_emoji = persona.linguistic_style['emoji_freq']
    if expected_emoji == 0 and emoji_count > 2:
        return 0.7  # Penalize excessive emojis
    elif expected_emoji > 5 and emoji_count < 3:
        return 0.8  # Penalize lack of emojis
    tone = persona.linguistic_style['tone']
    if tone == 'formal' and any(word in tweet_text.lower() for word in ['lol', 'omg', 'wtf']):
        return 0.6
    if tone == 'humorous' and quick_sentiment(tweet_text) < 0.1:
        return 0.7
    return 1.0  # No penalty


def simulate_persona_engagement(
    tweet_text,
    persona,
    debug=False,
    *,
    tweet_lower=None,
    words=None,
    phrases=None,
    all_content=None,
    sentiment=None,
):
    if debug:
        print(f"\n=== Debug: Persona {persona.id} ===")
        print(f"Interests: {list(persona.interests)[:5]}...")  # Show first 5
        print(f"Triggers: {list(persona.triggers)[:5]}...")    # Show first 5
        print(f"Engagement style: {persona.engagement_style}")

    # Use precomputed features if provided (from parallel_engagement); otherwise compute here.
    if tweet_lower is None:
        tweet_lower = tweet_text.lower()
    if words is None:
        words = {word.strip('.,!?":;').lower() for word in tweet_text.split()}
    if phrases is None:
        phrases = {phrase.strip().lower() for phrase in _SPLIT_RE.split(tweet_text) if phrase.strip()}
    # Get dynamic base engagement rates based on persona type
    base_engagement = {
        'lurker': {'like': 0.01, 'retweet': 0.002, 'reply': 0.001},
        'casual': {'like': 0.02, 'retweet': 0.005, 'reply': 0.003},
        'average': {'like': 0.03, 'retweet': 0.01, 'reply': 0.005},
        'proactive': {'like': 0.05, 'retweet': 0.02, 'reply': 0.01},
        'influencer': {'like': 0.08, 'retweet': 0.03, 'reply': 0.015}
    }.get(getattr(persona, 'engagement_style', 'average'), {'like': 0.03, 'retweet': 0.01, 'reply': 0.005})
    if all_content is None:
        all_content = words | phrases
    if sentiment is None:
        sentiment = quick_sentiment(tweet_text)

    # Check for dealbreakers and pet peeves (use pre-lowered text)
    low = tweet_lower or tweet_text.lower()
    if any(peeve in low for peeve in persona.pet_peeves) or any(db in low for db in persona.dealbreakers):
        if debug:
            _log("❌ Dealbreaker/pet peeve detected - zero engagement")
        return {'like': 0, 'retweet': 0, 'reply': 0}

    # Enhanced content-type matching with fuzzy logic
    content_match = 1.0
    exact_matches = sum(1 for topic in persona.content_types if topic in low)
    if exact_matches > 0:
        content_match = 1.0 + (exact_matches * 0.3)  # 30% boost per match

    # Fuzzy content matching for partial matches (RapidFuzz)
    # Cap fuzzy work (protect against worst-case loops)
    sample_words = words if len(words) <= 80 else set(list(words)[:80])
    partial_matches = 0
    for content_type in persona.content_types:
        for word in sample_words:
            try:
                if _partial_ratio(content_type, word) > 70:
                    partial_matches += 1
                    break
            except:
                # Fallback to basic substring matching
                if content_type in word:
                    partial_matches += 1
                    break
    if partial_matches > 0:
        content_match = max(content_match, 1.0 + (partial_matches * 0.2))  # 20% boost per partial match

    # Emotional trigger matching
    emotion = analyze_emotion(tweet_text)
    emotion_match = 1.0
    if persona.emotional_triggers:
        if emotion in persona.emotional_triggers:
            emotion_match = 1.4  # Strong boost for matching emotions
        else:
            emotion_match = 0.7  # Slight penalty for non-matching emotions

    # Enhanced interest/trigger matching
    interest_overlap = len(all_content & persona.interests)
    trigger_overlap = len(all_content & persona.triggers)
    
    interest_match = min(1.5, 1.0 + (interest_overlap * 0.2))  # Cap at 50% boost
    trigger_match = min(2.0, 1.0 + (trigger_overlap * 0.3))   # Cap at 100% boost

    # Style matching
    style_match = match_style(tweet_text, persona)

    # Engagement style modifier
    style_modifier = {
        'lurker': 0.5,
        'casual': 0.8,
        'average': 1.0,
        'proactive': 1.3,
        'influencer': 1.5
    }.get(persona.engagement_style, 1.0)

    if debug:
        _log(f"Content match: {content_match:.2f}")
        _log(f"Emotion match: {emotion_match:.2f}")
        _log(f"Interest match: {interest_match:.2f} (overlaps: {interest_overlap})")
        _log(f"Trigger match: {trigger_match:.2f} (overlaps: {trigger_overlap})")
        _log(f"Style modifier: {style_modifier:.2f}")

    # Realistic engagement probability calculation
    like_prob = (
        base_engagement['like'] 
        * interest_match 
        * trigger_match
        * content_match
        * emotion_match
        * style_match
        * style_modifier
        * persona.follower_weight
    )
    like_prob = min(like_prob, 0.85)  # Cap at 85% max engagement
    
    retweet_prob = min(
        like_prob * 0.6,  # Retweets are typically 60% of likes
        base_engagement['retweet'] * trigger_match * content_match * style_modifier
    )
    
    # Reply probability calculation
    question_boost = 1.8 if any(q in tweet_text for q in ['?', 'what', 'how', 'why', 'thoughts', 'opinion']) else 1.0
    controversy_boost = 1.5 if emotion == 'negative' or trigger_match > 1.3 else 1.0
    style_boost = 1.6 if persona.engagement_style == 'proactive' else 1.0
    
    reply_prob = (
        base_engagement['reply']
        * interest_match
        * trigger_match
        * question_boost
        * controversy_boost
        * style_boost
        * persona.follower_weight
    )
    reply_prob = min(reply_prob, 0.75)  # Cap at 75%
    
    final_engagement = {
        'like': like_prob,
        'retweet': retweet_prob,
        'reply': reply_prob,
        'match_factors': {
            'content': content_match,
            'emotion': emotion_match,
            'interests': interest_match,
            'triggers': trigger_match,
            'style': style_match,
            'question_boost': question_boost,
            'controversy_boost': controversy_boost
        }
    }
    
    if debug:
        _log(f"Final engagement: L:{like_prob:.3f} RT:{retweet_prob:.3f} R:{reply_prob:.3f}")
    
    return final_engagement


def aggregate_engagement(tweet_text, personas, demographic_weights=None):
    """Runs full simulation with demographic insights"""
    total_likes = 0
    total_rts = 0
    total_replies = 0
    demographic_engagement = {}
    segment_insights = {}
    
    # Stop double-simulating inside aggregate_engagement
    per_persona = []  # (persona, engagement)

    for persona in personas:
        eng = simulate_persona_engagement(tweet_text, persona)
        per_persona.append((persona, eng))
        weight = 1.0 if not demographic_weights else demographic_weights.get(persona.id, 1.0)
        total_likes += eng['like'] * weight
        total_rts += eng['retweet'] * weight
        total_replies += eng['reply'] * weight

        # Record demographic insights with better key extraction
        demo_data = persona.demographic
        job = demo_data.get('job', demo_data.get('occupation', demo_data.get('profession', 'Unknown')))
        age = demo_data.get('age', demo_data.get('age_range', ''))
        demo_key = f"{job}-{age}"
        if demo_key not in demographic_engagement:
            demographic_engagement[demo_key] = {
                'count': 0,
                'total_likes': 0,
                'total_rts': 0,
                'total_replies': 0
            }
        demographic_engagement[demo_key]['count'] += 1
        demographic_engagement[demo_key]['total_likes'] += eng['like']
        demographic_engagement[demo_key]['total_rts'] += eng['retweet']
        demographic_engagement[demo_key]['total_replies'] += eng['reply']

        # Record high-engagement personas (lowered thresholds)
        if eng['like'] > 0.02 or eng['retweet'] > 0.008 or eng['reply'] > 0.005:
            segment = job  # Use the extracted job/occupation
            if segment not in segment_insights:
                segment_insights[segment] = {
                    'count': 0,
                    'avg_engagement': 0,
                    'match_factors': {}
                }
            segment_insights[segment]['count'] += 1
            segment_insights[segment]['avg_engagement'] = (
                segment_insights[segment]['avg_engagement'] * 
                (segment_insights[segment]['count'] - 1) + 
                eng['like']
            ) / segment_insights[segment]['count']
            
            # Only add match_factors if they exist
            if 'match_factors' in eng:
                for factor, value in eng['match_factors'].items():
                    segment_insights[segment]['match_factors'][factor] = (
                        segment_insights[segment]['match_factors'].get(factor, 0) + value
                    )

    # Process segment insights
    top_segments = []
    for segment, data in segment_insights.items():
        avg_factors = {}
        for factor, total in data['match_factors'].items():
            avg_factors[factor] = total / data['count']
        top_segments.append({
            'segment': segment,
            'engagement_score': data['avg_engagement'],
            'match_factors': avg_factors
        })
    top_segments.sort(key=lambda x: x['engagement_score'], reverse=True)

    demo_distribution = []
    for demo, data in demographic_engagement.items():
        demo_distribution.append({
            'demographic': demo,
            'avg_like': data['total_likes'] / data['count'],
            'percent_of_audience': data['count'] / len(personas)
        })

    # More realistic scaling
    scale_factor = 20000 / len(personas)  # Assume 20k potential reach

    # Reuse computed results instead of double-simulating
    sorted_personas = sorted(per_persona, key=lambda x: x[1]['like'], reverse=True)
    top_10_percent = sorted_personas[:max(1, int(len(sorted_personas) * 0.1))]

    # Add top 10% personas to insights with better demographic extraction
    for persona, engagement in top_10_percent:
        demo_data = persona.demographic
        job = demo_data.get('job', demo_data.get('occupation', demo_data.get('profession', 'Unknown')))
        age = demo_data.get('age', demo_data.get('age_range', ''))
        demo_key = f"{job}-{age}"
        if demo_key not in demographic_engagement:
            demographic_engagement[demo_key] = {
                'count': 0,
                'total_likes': 0,
                'total_rts': 0,
                'total_replies': 0
            }
        demographic_engagement[demo_key]['count'] += 1
        demographic_engagement[demo_key]['total_likes'] += engagement['like']
        demographic_engagement[demo_key]['total_rts'] += engagement['retweet']
        demographic_engagement[demo_key]['total_replies'] += engagement['reply']

    # For Elo tournament compatibility, add engaged_personas (lowered thresholds)
    engaged_personas = [
        {'id': p.id, 'demographic': p.demographic, 'engagement': e}
        for (p, e) in sorted_personas
        if e.get('like', 0) > 0.02 or e.get('retweet', 0) > 0.008 or e.get('reply', 0) > 0.005
    ][:20]
    
    return {
        'persona_likes': total_likes * scale_factor,
        'persona_rts': total_rts * scale_factor,
        'persona_replies': total_replies * scale_factor,
        'top_segments': top_segments[:5],
        'demographic_distribution': demo_distribution,
        'engaged_personas': engaged_personas
    }

# New Elo-based optimization system
def generate_tweet_variants(original_tweet, num_variants=4):
    variants = [original_tweet]
    strategies = [
        lambda t: t + "!" * random.randint(1, 3),
        lambda t: t.replace(".", "!!!") if random.random() > 0.5 else t,
        lambda t: t + " " + random.choice(["🔥", "🚀", "💯", "👏"]),
        lambda t: t.replace("you", "one") if random.random() > 0.7 else t,
        lambda t: re.sub(r'\b(\w+)\b', lambda m: m.group(1).title(), t),
        lambda t: t + "\n\n" + random.choice([
            "Thoughts?", "Agree?", "Let me know your opinion!",
            "Tag someone who needs to see this", "How would you improve this?"
        ]),
        lambda t: t + " " + random.choice([
            "#GrowthMindset", "#InvestWisely", "#TechInnovation",
            "#CareerAdvice", "#FinancialFreedom"
        ])
    ]
    for _ in range(num_variants):
        variant = original_tweet
        for _ in range(random.randint(1, 3)):
            variant = random.choice(strategies)(variant)
        variants.append(variant)
    return variants

def calculate_elo_prob(rating_a, rating_b):
    """Calculate win probability for player A against player B"""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_elo(winner_elo, loser_elo, k=ELO_K):
    """Update Elo ratings after a matchup"""
    prob_winner = calculate_elo_prob(winner_elo, loser_elo)
    prob_loser = calculate_elo_prob(loser_elo, winner_elo)
    new_winner_elo = winner_elo + k * (1 - prob_winner)
    new_loser_elo = loser_elo + k * (0 - prob_loser)
    return new_winner_elo, new_loser_elo

def run_elo_tournament(tweet_variants, personas, demographic_weights=None):
    """Run simulated tournament between tweet variants"""
    # Initialize ratings
    ratings = {tweet: BASE_ELO for tweet in tweet_variants}
    engagement_data = {}
    win_counts = {tweet: 0 for tweet in tweet_variants}
    # Pre-calculate engagement for efficiency - use parallel fast-path
    for tweet in tweet_variants:
        engagement_data[tweet] = parallel_engagement(
            tweet, personas, demographic_weights
        )
    # Run round-robin tournament
    for i, tweet_a in enumerate(tweet_variants):
        for tweet_b in tweet_variants[i+1:]:
            # Determine winner based on engagement
            if engagement_data[tweet_a]['persona_likes'] > engagement_data[tweet_b]['persona_likes']:
                winner, loser = tweet_a, tweet_b
            else:
                winner, loser = tweet_b, tweet_a
            # Update ratings
            new_winner_elo, new_loser_elo = update_elo(
                ratings[winner], ratings[loser]
            )
            ratings[winner] = new_winner_elo
            ratings[loser] = new_loser_elo
            win_counts[winner] += 1
    # Prepare insights
    insights = []
    for tweet in tweet_variants:
        top_demographics = {}
        for persona in engagement_data[tweet]['engaged_personas']:
            # Get demographic info safely
            demo_data = persona.get('demographic', {})
            demo = (demo_data.get('segment') or 
                   demo_data.get('job') or 
                   demo_data.get('occupation') or 
                   f"{demo_data.get('age', 'Unknown')}-{demo_data.get('job', 'Unknown')}")
            top_demographics[demo] = top_demographics.get(demo, 0) + 1
        insights.append({
            'text': tweet,
            'elo': ratings[tweet],
            'wins': win_counts[tweet],
            'persona_likes': engagement_data[tweet]['persona_likes'],
            'top_demographics': dict(sorted(
                top_demographics.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3])
        })
    # Sort by Elo descending
    return sorted(insights, key=lambda x: x['elo'], reverse=True)

def optimize_and_predict(original_tweet, personas, demographic_weights=None, num_variants=4):
    """Full optimization pipeline with explainable insights"""
    # Generate variants
    variants = generate_tweet_variants(original_tweet, num_variants)
    # Run Elo tournament
    tournament_results = run_elo_tournament(variants, personas, demographic_weights)
    # Prepare output
    best_variant = tournament_results[0]
    worst_variant = tournament_results[-1]
    # Explainability analysis
    improvement = best_variant['persona_likes'] - tournament_results[1]['persona_likes']
    denom = tournament_results[1]['persona_likes']
    if denom == 0:
        improvement_pct = 0.0
    else:
        improvement_pct = (improvement / denom) * 100
    return {
        'best_variant': best_variant,
        'all_variants': tournament_results,
        'improvement_analysis': {
            'expected_likes_improvement': improvement,
            'percent_improvement': f"{improvement_pct:.1f}%",
            'key_demographics': list(best_variant['top_demographics'].keys()),
            'comparison': f"Beats original by {best_variant['elo'] - tournament_results[1]['elo']:.0f} Elo points"
        },
        'optimization_insights': [
            f"Top variant resonates with {len(best_variant['top_demographics'])} key demographics",
            f"Worst variant: '{worst_variant['text'][:30]}...' scored {worst_variant['elo']:.0f}",
            f"Tournament decided by {len(variants)*(len(variants)-1)//2} simulated matchups"
        ]
    }

# Dynamic emoji system
EMOJI_CATEGORIES = {
    'money': ['💸', '🤑', '💰', '🏦', '💵', '💲', '🪙'],
    'success': ['🌟', '🏅', '🎖️', '🚀', '🎉', '👑', '🧠'],
    'network': ['🤝', '🌐', '🗣️', '👥', '📞'],
    'skills': ['🛠️', '📚', '💡', '🧑‍💻', '🖥️', '📱'],
    'crypto': ['🪙', '💎', '📈', '🤑', '💻'],
    'health': ['💪', '🏋️', '🥗', '🧘', '🏃'],
    'confidence': ['😎', '💯', '🔥', '🌟'],
    'misc': ['✨', '🌈', '🎊', '🎆', '🎇']
}

def pick_emojis(category, n=2):
    return ''.join(random.sample(EMOJI_CATEGORIES.get(category, EMOJI_CATEGORIES['misc']), n))

class PersonaEngineAuditor:
    def __init__(self, personas_path="personas.json"):
        self.personas = self._load_personas(personas_path)
        self.audit_data = []
        
    def _load_personas(self, path):
        """Load persona data from JSON"""
        create_sample_personas_if_missing(path)
        try:
            with open(path) as f:
                return json.load(f)
        except FileNotFoundError:
            _log(f"Warning: {path} not found. Using empty persona list.")
            return []
        except json.JSONDecodeError:
            _log(f"Warning: {path} contains invalid JSON. Using empty persona list.")
            return []
    
    def log_predictions(self, tweet_text, actual_engagement=None):
        """Record predictions vs actuals for analysis"""
        # Load personas using the existing load_personas function
        personas = load_personas("personas.json")
        prediction = aggregate_engagement(tweet_text, personas)
        
        # Debug: Print engaged personas info
        engaged_personas = prediction.get('engaged_personas', [])
        _log(f"DEBUG: Found {len(engaged_personas)} engaged personas for tweet: {tweet_text[:50]}...")
        if len(engaged_personas) > 0:
            _log(f"DEBUG: First engaged persona: ID={engaged_personas[0].get('id')}, "
                  f"Like={engaged_personas[0].get('engagement', {}).get('like', 0):.3f}")
        
        # Debug: Print top segments info
        top_segments = prediction.get('top_segments', [])
        _log(f"DEBUG: Found {len(top_segments)} top segments")
        if len(top_segments) > 0:
            _log(f"DEBUG: First segment: {top_segments[0].get('segment', 'Unknown')} "
                  f"with engagement {top_segments[0].get('engagement_score', 0):.3f}")
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "tweet_text": tweet_text,
            "predicted": {
                "likes": prediction['persona_likes'],
                "retweets": prediction['persona_rts'],
                "replies": prediction['persona_replies']
            },
            "actual": actual_engagement or {},
            "engaged_personas_count": len(engaged_personas),
            "top_segments": top_segments[:3]
        }
        
        self.audit_data.append(record)
        return record
    
    def analyze_consistency(self):
        """Calculate variance in predictions"""
        if not self.audit_data:
            return {"error": "No audit data available"}
            
        predictions = [record["predicted"] for record in self.audit_data]
        
        likes = [p["likes"] for p in predictions]
        retweets = [p["retweets"] for p in predictions]
        replies = [p["replies"] for p in predictions]
        
        stats = {
            "likes_range": max(likes) - min(likes) if likes else 0,
            "retweets_range": max(retweets) - min(retweets) if retweets else 0,
            "replies_range": max(replies) - min(replies) if replies else 0,
            "likes_std": sum([(x - sum(likes)/len(likes))**2 for x in likes])**0.5 / len(likes) if likes else 0,
            "retweets_std": sum([(x - sum(retweets)/len(retweets))**2 for x in retweets])**0.5 / len(retweets) if retweets else 0,
            "replies_std": sum([(x - sum(replies)/len(replies))**2 for x in replies])**0.5 / len(replies) if replies else 0,
            "total_predictions": len(self.audit_data)
        }
        
        return stats
    
    def audit_inputs(self):
        """Analyze what variables the engine considers"""
        input_factors = defaultdict(int)
        
        for persona in self.personas:
            # Count how often each feature appears in personas
            if "follower_count" in persona.get("behavior", {}):
                input_factors["follower_count"] += 1
            if "interests" in persona.get("psychographics", {}):
                input_factors["content_interests"] += 1
            if "triggers" in persona.get("psychographics", {}):
                input_factors["content_triggers"] += 1
            if "emotional_triggers" in persona.get("engagement", {}):
                input_factors["emotional_triggers"] += 1
            if "dealbreakers" in persona.get("engagement", {}):
                input_factors["dealbreakers"] += 1
            if "pet_peeves" in persona.get("behavior", {}):
                input_factors["pet_peeves"] += 1
            if "content_type" in persona.get("behavior", {}):
                input_factors["content_types"] += 1
        
        # Add total personas for context
        input_factors["total_personas"] = len(self.personas)
        
        return dict(input_factors)
    
    def identify_bottlenecks(self):
        """Find why outputs are similar"""
        bottlenecks = []
        
        # Check if we have enough data
        if len(self.audit_data) < 3:
            bottlenecks.append("Insufficient test data - need at least 3 diverse tweets")
            return bottlenecks
        
        input_audit = self.audit_inputs()
        total_personas = input_audit.get("total_personas", 1)
        
        # 1. Check if follower count dominates
        if input_audit.get("follower_count", 0) / total_personas > 0.8:
            bottlenecks.append("Over-reliance on follower count")
        
        # 2. Check for missing NLP features
        if input_audit.get("content_interests", 0) / total_personas < 0.3:
            bottlenecks.append("Lack of content-aware features (interests)")
            
        if input_audit.get("emotional_triggers", 0) / total_personas < 0.3:
            bottlenecks.append("Lack of emotional trigger features")
            
        # 3. Check prediction ranges
        stats = self.analyze_consistency()
        if stats.get("likes_range", 0) < 1000:
            bottlenecks.append("Overly consistent like predictions (low variance)")
            
        if stats.get("retweets_range", 0) < 500:
            bottlenecks.append("Overly consistent retweet predictions (low variance)")
            
        # 4. Check for engaged personas variation
        engaged_counts = [record.get("engaged_personas_count", 0) for record in self.audit_data]
        if max(engaged_counts) - min(engaged_counts) < 3:
            bottlenecks.append("Similar engaged persona counts across different tweets")
            
        return bottlenecks or ["No obvious bottlenecks found"]
    
    def analyze_prediction_accuracy(self):
        """Analyze prediction accuracy where actual data is available"""
        accuracy_stats = {
            "predictions_with_actuals": 0,
            "avg_like_error": 0,
            "avg_retweet_error": 0,
            "avg_reply_error": 0
        }
        
        predictions_with_actuals = [
            record for record in self.audit_data 
            if record["actual"] and "likes" in record["actual"]
        ]
        
        if not predictions_with_actuals:
            return accuracy_stats
            
        accuracy_stats["predictions_with_actuals"] = len(predictions_with_actuals)
        
        like_errors = []
        retweet_errors = []
        reply_errors = []
        
        for record in predictions_with_actuals:
            pred = record["predicted"]
            actual = record["actual"]
            
            if "likes" in actual:
                like_errors.append(abs(pred["likes"] - actual["likes"]))
            if "retweets" in actual:
                retweet_errors.append(abs(pred["retweets"] - actual["retweets"]))
            if "replies" in actual:
                reply_errors.append(abs(pred["replies"] - actual["replies"]))
        
        if like_errors:
            accuracy_stats["avg_like_error"] = sum(like_errors) / len(like_errors)
        if retweet_errors:
            accuracy_stats["avg_retweet_error"] = sum(retweet_errors) / len(retweet_errors)
        if reply_errors:
            accuracy_stats["avg_reply_error"] = sum(reply_errors) / len(reply_errors)
            
        return accuracy_stats
    
    def generate_report(self, output_file="persona_audit_report.json"):
        """Compile all diagnostics"""
        report = {
            "audit_timestamp": datetime.now().isoformat(),
            "summary_stats": self.analyze_consistency(),
            "input_factors": self.audit_inputs(),
            "bottlenecks": self.identify_bottlenecks(),
            "accuracy_analysis": self.analyze_prediction_accuracy(),
            "sample_predictions": self.audit_data[:5],  # First 5 records
            "total_tests": len(self.audit_data)
        }
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
            
        return report
    
    def quick_test(self, diverse_tweets=None):
        """Run a quick test with diverse tweet types"""
        if diverse_tweets is None:
            diverse_tweets = [
                ("Just had the best coffee ever! ☕️ #MondayMotivation", {"likes": 42, "retweets": 3, "replies": 5}),
                ("BREAKING: New study shows remote work increases productivity by 47% 📈", {"likes": 2100, "retweets": 950, "replies": 180}),
                ("Check out my new blog post about financial freedom 💰", {"likes": 150, "retweets": 30, "replies": 12}),
                ("Why does nobody talk about the mental health crisis in tech? 🧠", {"likes": 380, "retweets": 120, "replies": 67}),
                ("Made $10k this month with my side hustle! Here's how... 🚀", {"likes": 890, "retweets": 245, "replies": 89}),
                ("Unpopular opinion: Most productivity advice is just procrastination in disguise", {"likes": 1250, "retweets": 340, "replies": 156})
            ]
        
        print("Running quick diagnostic test...")
        for text, actual in diverse_tweets:
            self.log_predictions(text, actual)
        
        report = self.generate_report()
        return report

def create_sample_personas_if_missing(path="personas.json"):
    """Create sample personas if the file doesn't exist"""
    import os
    if not os.path.exists(path):
        _log(f"Creating sample personas file at {path}")
        sample_personas = [
            {
                "id": 1,
                "demographics": {
                    "job": "Software Engineer",
                    "age": "25-34",
                    "occupation": "Developer",
                    "segment": "Tech Professional"
                },
                "psychographics": {
                    "interests": ["coding", "technology", "startups", "innovation"],
                    "triggers": ["tech", "programming", "software"]
                },
                "behavior": {
                    "content_type": ["educational", "tech"],
                    "follower_count": "1k-10k",
                    "pet_peeves": ["spam", "clickbait"]
                },
                "engagement": {
                    "style": "proactive",
                    "content_triggers": ["tech trends", "coding tips"],
                    "emotional_triggers": ["positive", "educational"],
                    "dealbreakers": ["hate speech", "misinformation"]
                },
                "linguistic": {
                    "emoji_freq": 2,
                    "tone": "professional"
                }
            },
            {
                "id": 2,
                "demographics": {
                    "job": "Marketing Manager",
                    "age": "30-39",
                    "occupation": "Marketer", 
                    "segment": "Business Professional"
                },
                "psychographics": {
                    "interests": ["marketing", "business", "growth", "networking"],
                    "triggers": ["business", "marketing", "growth"]
                },
                "behavior": {
                    "content_type": ["business", "marketing"],
                    "follower_count": "5k-15k",
                    "pet_peeves": ["poor grammar", "unprofessional"]
                },
                "engagement": {
                    "style": "casual",
                    "content_triggers": ["business tips", "marketing insights"],
                    "emotional_triggers": ["motivational", "success"],
                    "dealbreakers": ["inappropriate content"]
                },
                "linguistic": {
                    "emoji_freq": 3,
                    "tone": "friendly"
                }
            },
            {
                "id": 3,
                "demographics": {
                    "job": "Entrepreneur",
                    "age": "28-35",
                    "occupation": "Founder",
                    "segment": "Startup Founder"
                },
                "psychographics": {
                    "interests": ["entrepreneurship", "business", "investing", "freedom"],
                    "triggers": ["startup", "business", "entrepreneur", "money"]
                },
                "behavior": {
                    "content_type": ["inspirational", "business"],
                    "follower_count": "10k-50k",
                    "pet_peeves": ["negativity", "time wasting"]
                },
                "engagement": {
                    "style": "influencer",
                    "content_triggers": ["success stories", "business advice"],
                    "emotional_triggers": ["motivational", "inspiring"],
                    "dealbreakers": ["scams", "get rich quick"]
                },
                "linguistic": {
                    "emoji_freq": 4,
                    "tone": "motivational"
                }
            }
        ]
        
        with open(path, 'w') as f:
            json.dump(sample_personas, f, indent=2)
        _log(f"Created {len(sample_personas)} sample personas")
        return sample_personas
    return None

if __name__ == "__main__":
    # Create sample personas if missing
    create_sample_personas_if_missing("personas.json")
    
    # Load personas
    personas = load_personas("personas.json")
    # Example tweet
    tweet = """2018: 84-hour weeks as an electrician.
2020: Minimum wage at Dominos.
2022: $47K month business while traveling the world.

If you want to learn the best 'freedom' business to start in 2025, read this:"""
    
    print("="*60)
    print("ORIGINAL PERSONA ENGINE")
    print("="*60)
    
    # Run parallel engagement
    result = parallel_engagement(tweet, personas)
    print("Parallel Engagement Result:")
    print(result)
    # Run optimization and Elo tournament
    opt_result = optimize_and_predict(tweet, personas)
    print("\nOptimization & Elo Tournament Result:")
    print(opt_result)
    
    print("\n" + "="*60)
    print("ENHANCED PERSONA ENGINE")
    print("="*60)
    
    # Initialize enhanced engine
    class MockTopicModel:
        def __init__(self):
            # Comprehensive topic mapping with keyword patterns
            self.topic_mappings = {
                # Business & Entrepreneurship
                'business': ['business', 'entrepreneur', 'startup', 'company', 'corporate', 'profit', 'revenue', 'sales', 'marketing', 'strategy', 'growth', 'scaling', 'funding', 'investor', 'venture', 'capital', 'roi', 'kpi', 'metrics', 'analytics'],
                'finance': ['money', 'invest', 'investment', 'finance', 'financial', 'budget', 'savings', 'debt', 'credit', 'loan', 'mortgage', 'insurance', 'retirement', 'portfolio', 'stocks', 'bonds', 'crypto', 'bitcoin', 'ethereum', 'trading'],
                'entrepreneurship': ['founder', 'startup', 'venture', 'innovation', 'disrupt', 'scale', 'pivot', 'bootstrap', 'unicorn', 'ipo', 'acquisition', 'exit', 'valuation', 'pitch', 'seed', 'series'],
                
                # Technology
                'tech': ['technology', 'tech', 'digital', 'software', 'hardware', 'computer', 'programming', 'development', 'coding', 'algorithm', 'data', 'cloud', 'server', 'network', 'security', 'cybersecurity'],
                'ai': ['ai', 'artificial', 'intelligence', 'machine', 'learning', 'deep', 'neural', 'automation', 'chatgpt', 'openai', 'llm', 'gpt', 'model', 'training', 'algorithm'],
                'programming': ['code', 'coding', 'python', 'javascript', 'java', 'react', 'node', 'api', 'framework', 'library', 'github', 'git', 'developer', 'software', 'bug', 'debug'],
                'web3': ['blockchain', 'crypto', 'nft', 'defi', 'web3', 'decentralized', 'smart', 'contract', 'ethereum', 'bitcoin', 'token', 'dao', 'metaverse'],
                
                # Career & Professional
                'career': ['career', 'job', 'work', 'employment', 'interview', 'resume', 'cv', 'linkedin', 'networking', 'promotion', 'salary', 'skills', 'experience', 'professional'],
                'productivity': ['productivity', 'efficient', 'optimize', 'workflow', 'time', 'management', 'focus', 'goals', 'habits', 'routine', 'system', 'organization'],
                'leadership': ['leader', 'leadership', 'management', 'team', 'communication', 'delegation', 'motivation', 'culture', 'vision', 'strategy', 'mentoring'],
                
                # Lifestyle & Personal
                'health': ['health', 'fitness', 'workout', 'exercise', 'gym', 'nutrition', 'diet', 'wellness', 'mental', 'therapy', 'meditation', 'mindfulness', 'sleep'],
                'travel': ['travel', 'vacation', 'trip', 'adventure', 'explore', 'journey', 'destination', 'culture', 'nomad', 'remote', 'location'],
                'food': ['food', 'cooking', 'recipe', 'restaurant', 'chef', 'cuisine', 'meal', 'dinner', 'lunch', 'breakfast', 'coffee', 'wine'],
                'lifestyle': ['lifestyle', 'life', 'living', 'home', 'family', 'relationship', 'friend', 'social', 'hobby', 'passion', 'balance'],
                
                # Education & Learning
                'education': ['education', 'learning', 'school', 'university', 'course', 'study', 'knowledge', 'skill', 'training', 'certification', 'degree'],
                'self_improvement': ['improvement', 'growth', 'development', 'mindset', 'confidence', 'success', 'achievement', 'goal', 'discipline', 'motivation'],
                'books': ['book', 'reading', 'author', 'novel', 'literature', 'story', 'knowledge', 'learn', 'wisdom', 'education'],
                
                # Entertainment & Media
                'entertainment': ['movie', 'film', 'tv', 'show', 'series', 'netflix', 'streaming', 'actor', 'actress', 'director', 'cinema'],
                'music': ['music', 'song', 'album', 'artist', 'band', 'concert', 'festival', 'spotify', 'playlist', 'genre'],
                'gaming': ['game', 'gaming', 'video', 'console', 'pc', 'mobile', 'esports', 'streamer', 'twitch', 'youtube'],
                'social_media': ['social', 'media', 'instagram', 'twitter', 'facebook', 'tiktok', 'youtube', 'influencer', 'content', 'viral'],
                
                # Science & Innovation
                'science': ['science', 'research', 'study', 'experiment', 'discovery', 'innovation', 'technology', 'medicine', 'biology', 'physics', 'chemistry'],
                'environment': ['environment', 'climate', 'sustainability', 'green', 'renewable', 'energy', 'carbon', 'pollution', 'conservation', 'nature'],
                'space': ['space', 'nasa', 'rocket', 'mars', 'moon', 'satellite', 'astronaut', 'universe', 'galaxy', 'spacex'],
                
                # Politics & Society
                'politics': ['politics', 'government', 'policy', 'election', 'vote', 'democracy', 'law', 'regulation', 'congress', 'senate'],
                'society': ['society', 'community', 'culture', 'social', 'equality', 'justice', 'rights', 'diversity', 'inclusion', 'humanity'],
                'news': ['news', 'breaking', 'update', 'report', 'journalist', 'media', 'press', 'current', 'events', 'headline'],
                
                # Sports & Recreation
                'sports': ['sports', 'football', 'basketball', 'soccer', 'baseball', 'tennis', 'golf', 'olympics', 'athlete', 'competition'],
                'recreation': ['hobby', 'fun', 'activity', 'recreation', 'leisure', 'weekend', 'relax', 'enjoy', 'entertainment'],
                
                # Emerging Topics
                'metaverse': ['metaverse', 'virtual', 'reality', 'vr', 'ar', 'augmented', 'immersive', 'avatar', 'digital', 'world'],
                'sustainability': ['sustainable', 'eco', 'green', 'renewable', 'clean', 'environmental', 'carbon', 'neutral', 'climate'],
                'remote_work': ['remote', 'work', 'home', 'distributed', 'flexible', 'digital', 'nomad', 'virtual', 'online'],
                'mental_health': ['mental', 'health', 'wellness', 'anxiety', 'depression', 'stress', 'therapy', 'mindfulness', 'self-care']
            }
            
            # Create reverse mapping for faster lookup
            self.keyword_to_topics = {}
            for topic, keywords in self.topic_mappings.items():
                for keyword in keywords:
                    if keyword not in self.keyword_to_topics:
                        self.keyword_to_topics[keyword] = []
                    self.keyword_to_topics[keyword].append(topic)
        
        def predict(self, text):
            """Predict topics based on text content with sophisticated matching"""
            text_lower = text.lower()
            found_topics = set()
            
            # Direct keyword matching
            for keyword, topics in self.keyword_to_topics.items():
                if keyword in text_lower:
                    found_topics.update(topics)
            
            # Pattern-based matching for better coverage
            patterns = {
                'business': r'\b(startup|entrepreneur|business|profit|revenue|sales)\b',
                'tech': r'\b(ai|tech|software|code|programming|digital)\b',
                'finance': r'\b(money|invest|crypto|bitcoin|trading|portfolio)\b',
                'career': r'\b(job|career|work|professional|interview|resume)\b',
                'health': r'\b(health|fitness|workout|wellness|mental)\b',
                'education': r'\b(learn|education|course|skill|training)\b',
                'lifestyle': r'\b(life|lifestyle|travel|food|family)\b',
                'social_media': r'\b(viral|influencer|content|social|media)\b'
            }
            
            for topic, pattern in patterns.items():
                if re.search(pattern, text_lower):
                    found_topics.add(topic)
            
            # Emoji-based topic detection
            emoji_topics = {
                '💰💸🤑💰🏦💵💲🪙': 'finance',
                '🚀🌟⭐✨🔥💯': 'success',
                '💻🖥️📱⌚🎮': 'tech',
                '📚📖✏️📝🎓': 'education',
                '💪🏋️‍♂️🏃‍♀️🧘‍♂️🥗': 'health',
                '✈️🏖️🗺️🎒🌍': 'travel',
                '🎬🎵🎮📺🎪': 'entertainment'
            }
            
            for emojis, topic in emoji_topics.items():
                if any(emoji in text for emoji in emojis):
                    found_topics.add(topic)
            
            # Hashtag analysis
            hashtags = re.findall(r'#(\w+)', text_lower)
            for hashtag in hashtags:
                # Check if hashtag matches any keyword
                if hashtag in self.keyword_to_topics:
                    found_topics.update(self.keyword_to_topics[hashtag])
                
                # Common hashtag patterns
                hashtag_patterns = {
                    'business': ['entrepreneur', 'startup', 'business', 'hustle', 'success'],
                    'tech': ['tech', 'ai', 'coding', 'programming', 'innovation'],
                    'finance': ['investing', 'crypto', 'money', 'wealth', 'finance'],
                    'motivation': ['motivation', 'inspiration', 'mindset', 'goals', 'growth'],
                    'lifestyle': ['lifestyle', 'life', 'wellness', 'balance', 'happiness']
                }
                
                for topic, patterns in hashtag_patterns.items():
                    if any(pattern in hashtag for pattern in patterns):
                        found_topics.add(topic)
            
            # Question patterns suggest engagement topics
            if any(q in text_lower for q in ['?', 'thoughts', 'opinion', 'what do you think', 'agree']):
                found_topics.add('engagement')
            
            # News patterns
            if any(word in text_lower for word in ['breaking', 'news', 'update', 'report', 'announced']):
                found_topics.add('news')
            
            # Convert to list and ensure we always return at least one topic
            result = list(found_topics) if found_topics else ['general']
            
            # Limit to top 5 most relevant topics to avoid oversaturation
            return result[:5]
    
    enhanced_engine = EnhancedPersonaEngine({
        'topic': MockTopicModel()
    })
    
    # Set some trending hashtags for demonstration
    enhanced_engine.set_trending_hashtags(['entrepreneur', 'business', 'startup', 'freedom'])
    
    # Test diverse tweets with enhanced engine to showcase improved topic detection
    test_tweets = [
        ("Just had the best coffee ever! ☕️ #MondayMotivation", "casual"),
        ("BREAKING: New study shows remote work increases productivity by 47% 📈", "regular"),
        ("Check out my new blog post about financial freedom 💰", "regular"),
        ("Why does nobody talk about the mental health crisis in tech? 🧠", "influencer"),
        ("Made $10k this month with my side hustle! Here's how... 🚀 #entrepreneur", "regular"),
        ("Unpopular opinion: Most productivity advice is just procrastination in disguise", "influencer"),
        
        # Additional diverse test cases for enhanced topic detection
        ("Learning Python programming has changed my career trajectory completely! 💻 #coding", "regular"),
        ("Just finished my morning workout routine 💪 feeling energized for the day ahead", "casual"),
        ("AI and machine learning are revolutionizing healthcare diagnosis 🏥 #innovation", "influencer"),
        ("Crypto market volatility is teaching me patience 📊 #bitcoin #investing", "regular"),
        ("Reading a fascinating book about space exploration 🚀 #nasa #mars", "casual"),
        ("Working remotely from Bali this month 🏝️ digital nomad life is amazing!", "influencer"),
        ("New Netflix series recommendations? Need something binge-worthy 📺", "casual"),
        ("Climate change solutions require global cooperation 🌍 #sustainability", "influencer"),
        ("Gaming setup finally complete! RTX 4090 is incredible 🎮 #pcgaming", "regular"),
        ("Meditation practice has improved my focus significantly 🧘‍♂️ #mindfulness", "regular"),
        ("Startup founder life: 70% debugging, 30% pretending to have it together 😅", "influencer"),
        ("Virtual reality training for surgeons showing promising results 🥽 #medical", "regular"),
        ("Food truck business idea: Korean-Mexican fusion 🌮 thoughts?", "casual"),
        ("Blockchain technology beyond crypto: supply chain transparency 🔗", "influencer"),
        ("Olympic training requires 4am starts but dreams are worth it 🏅 #sports", "regular")
    ]
    
    print("\n🚀 ENHANCED ENGINE PREDICTIONS (with debug):")
    for i, (tweet_text, user_tier) in enumerate(test_tweets):
        user_id = f"user_{i+1}"
        prediction = enhanced_engine.predict_virality(tweet_text, user_id=user_id, user_tier=user_tier, debug=(i == 0))  # Debug first tweet
        
        print(f"\n--- Tweet {i+1} ({user_tier}) ---")
        print(f"Text: {tweet_text[:60]}...")
        print(f"Predictions: Likes: {prediction['predictions']['likes']:.0f}, "
              f"RTs: {prediction['predictions']['retweets']:.0f}, "
              f"Replies: {prediction['predictions']['replies']:.0f}")
        print(f"Viral Score: {prediction['viral_score']:.2f}")
        print(f"Topics: {prediction['content_features']['topics']}")
        print(f"Sentiment: {prediction['content_features']['sentiment']:.2f}")
        print(f"Feedback Adj: {prediction.get('feedback_adjustment', 1.0):.2f}")
    
    # Test individual persona engagement with debug
    print("\n🔍 INDIVIDUAL PERSONA DEBUG (First 3 personas):")
    test_personas = personas[:3]  # Test first 3 personas
    for persona in test_personas:
        engagement = simulate_persona_engagement(tweet, persona, debug=True)
    
    # Demonstrate enhanced feedback learning
    print("\n🎯 ENHANCED FEEDBACK LEARNING:")
    user_id = "demo_user"
    
    # Initial prediction
    initial_pred = enhanced_engine.predict_virality(tweet, user_id=user_id, user_tier="regular")
    print(f"Initial prediction: {initial_pred['predictions']['likes']:.0f} likes, {initial_pred['predictions']['retweets']:.0f} RTs")
    
    # Simulate multiple rounds of feedback
    test_actuals = [
        {'likes': 89, 'retweets': 12, 'replies': 4},    # Much lower than predicted
        {'likes': 134, 'retweets': 23, 'replies': 7},   # Still lower
        {'likes': 67, 'retweets': 8, 'replies': 2}      # Even lower
    ]
    
    for i, actual in enumerate(test_actuals):
        enhanced_engine.update_feedback(user_id, actual, initial_pred['predictions'])
        new_pred = enhanced_engine.predict_virality(tweet, user_id=user_id, user_tier="regular")
        print(f"After feedback {i+1}: {new_pred['predictions']['likes']:.0f} likes (adj: {new_pred.get('feedback_adjustment', 1.0):.2f})")
    
    # Test calibration
    print(f"\n⚖️ CALIBRATION FACTORS:")
    scale_factors = enhanced_engine.calculate_scale_factor()
    print(f"Likes scale: {scale_factors['likes']:.2f}")
    print(f"Retweets scale: {scale_factors['retweets']:.2f}")
    print(f"Replies scale: {scale_factors['replies']:.2f}")
    print("(These should be updated with your actual dataset statistics)")
    
    # Run diagnostic audit on original engine
    print("\n" + "="*60)
    print("ORIGINAL ENGINE DIAGNOSTIC AUDIT")
    print("="*60)
    
    auditor = PersonaEngineAuditor()
    
    # Run quick test with diverse tweets
    report = auditor.quick_test()
    
    print(f"\n📊 AUDIT SUMMARY:")
    print(f"   Total Tests: {report['total_tests']}")
    print(f"   Likes Range: {report['summary_stats']['likes_range']:.0f}")
    print(f"   Retweets Range: {report['summary_stats']['retweets_range']:.0f}")
    print(f"   Replies Range: {report['summary_stats']['replies_range']:.0f}")
    
    print(f"\n🔍 INPUT FACTORS:")
    input_factors = report['input_factors']
    total_personas = input_factors.get('total_personas', 1)
    for factor, count in input_factors.items():
        if factor != 'total_personas':
            percentage = (count / total_personas) * 100
            print(f"   {factor}: {count}/{total_personas} ({percentage:.1f}%)")
    
    print(f"\n⚠️  IDENTIFIED BOTTLENECKS:")
    for bottleneck in report['bottlenecks']:
        print(f"   • {bottleneck}")
    
    if report['accuracy_analysis']['predictions_with_actuals'] > 0:
        print(f"\n🎯 ACCURACY ANALYSIS:")
        acc = report['accuracy_analysis']
        print(f"   Predictions with actuals: {acc['predictions_with_actuals']}")
        print(f"   Avg like error: {acc['avg_like_error']:.0f}")
        print(f"   Avg retweet error: {acc['avg_retweet_error']:.0f}")
        print(f"   Avg reply error: {acc['avg_reply_error']:.0f}")
    
    print(f"\n📄 Full report saved to: persona_audit_report.json")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR IMPROVEMENT")
    print("="*60)
    print("✅ Enhanced engine shows:")
    print("   • Content-aware predictions (topics, sentiment)")
    print("   • Viral boosting for trending content")
    print("   • Dynamic user-specific adjustments")
    print("   • Feedback learning mechanism")
    print("   • Topic decay to prevent overuse")
    print("\n🔧 Next steps:")
    print("   • Replace mock topic model with real ML model")
    print("   • Integrate live trending hashtag data")
    print("   • Collect actual engagement data for feedback")
    print("   • Add more sophisticated NLP features")
    print("="*60)

    def test_enhanced_topic_model():
        """Test function to showcase the enhanced MockTopicModel capabilities"""
        print("Testing Enhanced MockTopicModel Topic Detection:")
        print("=" * 60)
        
        topic_model = MockTopicModel()
        
        # Test cases that showcase diverse topic detection
        test_cases = [
            "Building my startup with AI and machine learning 🚀 #entrepreneur",
            "Remote work productivity tips for developers 💻 #coding #productivity",
            "Crypto portfolio diversification strategies 💰 #bitcoin #investing",
            "Mental health awareness in tech industry 🧠 #wellness #mindfulness",
            "Climate change solutions through renewable energy 🌍 #sustainability",
            "Virtual reality gaming experience review 🎮 #vr #gaming",
            "Space exploration breakthrough at NASA 🚀 #space #science",
            "Healthy meal prep for busy professionals 🥗 #health #nutrition",
            "Breaking: New education policy announced 📚 #news #education",
            "Olympic training motivation 🏅 #sports #fitness"
        ]
        
        for text in test_cases:
            topics = topic_model.predict(text)
            print(f"Text: {text[:50]}...")
            print(f"Topics: {topics}")
            print("-" * 40)

    print("\n" + "="*60)
    print("ENHANCED TOPIC MODEL TEST")
    print("="*60)
    test_enhanced_topic_model()

    # Test the enhanced engine with diverse tweets
    print("\n📊 ENHANCED ENGINE TESTING WITH DIVERSE TOPICS:")
    print("=" * 60)
    
    for i, (tweet_text, persona_type) in enumerate(test_tweets):
        print(f"\n🐦 Tweet {i+1}: {tweet_text}")
        print(f"👤 Persona Type: {persona_type}")
        
        # Get topic prediction from enhanced model
        topics = enhanced_engine.ml_models['topic'].predict(tweet_text)
        print(f"🏷️ Detected Topics: {topics}")
        
        result = enhanced_engine.predict_virality(tweet_text, user_id=f"test_user_{i}", user_tier=persona_type)
        
        print(f"📈 Predictions:")
        print(f"   👍 Likes: {result['predictions']['likes']:.0f}")
        print(f"   🔄 Retweets: {result['predictions']['retweets']:.0f}")
        print(f"   💬 Replies: {result['predictions']['replies']:.0f}")
        print(f"   🔥 Viral Score: {result['viral_score']:.3f}")
    print(f"   🔁 Decay: {result['decay_factor']:.2f}")
    
    if 'engaged_personas' in result:
        print(f"   👥 Engaged Personas: {len(result['engaged_personas'])}")
    if 'top_segments' in result:
        print(f"   🎯 Top Segments: {result['top_segments'][:3]}")
    
    print("-" * 50)
