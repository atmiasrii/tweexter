# trending_hashtags.py
import os
import re
from datetime import datetime, timedelta
from typing import Set, List, Optional

# Global cache for trending hashtags
_cached_trending = None
_cache_timestamp = None
_cache_duration_hours = 24  # Refresh every 24 hours

def _load_trending_hashtags() -> Set[str]:
    """Load trending hashtags from file or use defaults"""
    # Try to load from file first
    trends_file = os.path.join(os.path.dirname(__file__), "..", "ab", "us_weekly_trends.txt")
    if os.path.exists(trends_file):
        try:
            with open(trends_file, 'r', encoding='utf-8') as f:
                trends = set()
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract hashtags from the line
                        hashtags = re.findall(r'#(\w+)', line.lower())
                        trends.update(hashtags)
                        # Also add the line itself if it looks like a hashtag/trend
                        if line.startswith('#'):
                            trends.add(line[1:].lower())
                        elif ' ' not in line and len(line) > 2:
                            trends.add(line.lower())
                if trends:
                    return trends
        except Exception as e:
            print(f"⚠️ Failed to load trends from {trends_file}: {e}")
    
    # Fallback: Default trending topics/hashtags (common ones)
    return {
        'ai', 'crypto', 'bitcoin', 'nft', 'web3', 'metaverse', 'blockchain',
        'covid', 'vaccine', 'pandemic', 'climate', 'ukraine', 'russia',
        'election', 'politics', 'trump', 'biden', 'news', 'breaking',
        'sports', 'nba', 'nfl', 'worldcup', 'olympics', 'football',
        'netflix', 'disney', 'marvel', 'starwars', 'gameofthrones',
        'iphone', 'android', 'tesla', 'spacex', 'apple', 'google',
        'music', 'grammys', 'oscars', 'emmys', 'fashion', 'style',
        'fitness', 'health', 'mental', 'wellness', 'selfcare',
        'travel', 'vacation', 'food', 'recipe', 'cooking',
        'gaming', 'esports', 'twitch', 'youtube', 'tiktok',
        'love', 'relationship', 'dating', 'wedding', 'family',
        'education', 'learning', 'career', 'job', 'work',
        'art', 'culture', 'book', 'reading', 'photography'
    }

def get_trending_hashtags() -> Set[str]:
    """Get current trending hashtags with caching"""
    global _cached_trending, _cache_timestamp
    
    now = datetime.now()
    
    # Check if cache is valid
    if (_cached_trending is not None and 
        _cache_timestamp is not None and 
        (now - _cache_timestamp).total_seconds() < _cache_duration_hours * 3600):
        return _cached_trending
    
    # Refresh cache
    _cached_trending = _load_trending_hashtags()
    _cache_timestamp = now
    return _cached_trending

def count_trending_hashtags(text: str) -> int:
    """
    Count how many trending hashtags are mentioned in the text.
    Returns integer count.
    """
    if not text:
        return 0
    
    trending = get_trending_hashtags()
    text_lower = text.lower()
    
    # Extract all hashtags from text
    hashtags_in_text = set(re.findall(r'#(\w+)', text_lower))
    
    # Also check for trending words/phrases without # symbol
    words_in_text = set(re.findall(r'\b\w+\b', text_lower))
    
    # Count matches
    count = 0
    count += len(hashtags_in_text.intersection(trending))
    count += len(words_in_text.intersection(trending))
    
    return count

def is_trending_topic(text: str, min_count: int = 1) -> bool:
    """Check if text contains trending topics/hashtags"""
    return count_trending_hashtags(text) >= min_count

def get_trending_words_in_text(text: str) -> List[str]:
    """Return list of trending words/hashtags found in text"""
    if not text:
        return []
    
    trending = get_trending_hashtags()
    text_lower = text.lower()
    
    found = []
    
    # Check hashtags
    hashtags_in_text = re.findall(r'#(\w+)', text_lower)
    found.extend([h for h in hashtags_in_text if h in trending])
    
    # Check words
    words_in_text = re.findall(r'\b\w+\b', text_lower)
    found.extend([w for w in words_in_text if w in trending])
    
    return list(set(found))  # Remove duplicates

# For debugging/testing
if __name__ == "__main__":
    test_texts = [
        "Just bought some #bitcoin and #ethereum! #crypto to the moon!",
        "Watching the new #Marvel movie on #Netflix tonight",
        "AI and machine learning are changing everything #AI #tech",
        "Regular tweet about my day nothing special here"
    ]
    
    for text in test_texts:
        count = count_trending_hashtags(text)
        words = get_trending_words_in_text(text)
        print(f"Text: {text}")
        print(f"Trending count: {count}")
        print(f"Trending words: {words}")
        print()
