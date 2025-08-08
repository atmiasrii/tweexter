# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import string
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from textstat import flesch_reading_ease, gunning_fog, smog_index
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
import en_core_web_sm
from collections import Counter

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
sentiment_analyzer = SentimentIntensityAnalyzer()
nlp = en_core_web_sm.load()

# Load your dataset
df = pd.read_csv('datazet.csv', parse_dates=['Timestamp'])

# Check dataset structure
print("Dataset columns:", df.columns.tolist())
print("Dataset shape:", df.shape)
print("First few rows of Tweet Text column:")
print(df['Tweet Text'].head())
print("Null values in Tweet Text:", df['Tweet Text'].isnull().sum())

# --- Step 1: Text Cleaning ---
def clean_tweet(text):
    # Handle missing or null values
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if it's not already
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove HTML entities
    text = re.sub(r'&\w+;', '', text)
    
    # Expand contractions
    contractions = {
        "don't": "do not", "doesn't": "does not", "isn't": "is not",
        "i'm": "i am", "we're": "we are", "you're": "you are",
        "can't": "cannot", "won't": "will not", "it's": "it is"
    }
    for cont, exp in contractions.items():
        text = text.replace(cont, exp)
    
    # Remove special characters except mentions, hashtags and emojis
    text = re.sub(r"[^\w\s@#!$%&*?:;'-]", '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

df['cleaned_text'] = df['Tweet Text'].apply(clean_tweet)

# --- Step 2: Feature Extraction ---
def extract_features(text):
    features = {}
    
    # Handle empty or null text
    if not text or not isinstance(text, str):
        text = ""
    
    # Structural Features
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['sentence_count'] = len(sent_tokenize(text)) if text else 0
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text else 0
    features['uppercase_ratio'] = sum(1 for char in text if char.isupper()) / len(text) if text else 0
    features['whitespace_ratio'] = text.count(' ') / len(text) if text else 0
    
    # Content Features (removed url_present and emoji_count - not in MODEL_FEATURES)
    features['hashtag_count'] = text.count('#')
    features['mention_count'] = text.count('@')
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['dollar_sign_count'] = text.count('$')
    
    # Readability Metrics
    try:
        features['flesch_reading_ease'] = flesch_reading_ease(text) if text else 0
        features['gunning_fog'] = gunning_fog(text) if text else 0
        features['smog_index'] = smog_index(text) if text else 0
    except:
        features['flesch_reading_ease'] = 0
        features['gunning_fog'] = 0
        features['smog_index'] = 0
    
    # Sentiment Analysis
    sentiment = sentiment_analyzer.polarity_scores(text)
    features['sentiment_neg'] = sentiment['neg']
    features['sentiment_neu'] = sentiment['neu']
    features['sentiment_pos'] = sentiment['pos']
    features['sentiment_compound'] = sentiment['compound']
    
    # Linguistic Features (using spaCy)
    doc = nlp(text)
    # POS tags
    pos_counts = Counter(token.pos_ for token in doc)
    for pos in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON']:
        features[f'{pos.lower()}_count'] = pos_counts.get(pos, 0)
    
    # Named Entities
    ner_counts = Counter(ent.label_ for ent in doc.ents)
    for entity in ['PERSON', 'ORG', 'GPE', 'PRODUCT']:
        features[f'{entity.lower()}_count'] = ner_counts.get(entity, 0)
    
    # Advanced Features
    features['quote_count'] = text.count('"') + text.count("'")
    features['unique_word_ratio'] = len(set(text.split())) / len(text.split()) if text else 0
    features['stopword_count'] = sum(1 for word in text.split() if word in stop_words)
    
    # Contextual Features
    features['first_word_uppercase'] = 1 if text and text[0].isupper() else 0
    features['has_hashtag_start'] = 1 if text.startswith('#') else 0
    
    # Temporal Features (extracted from timestamp in main df)
    
    return features

# Apply feature extraction
feature_df = df['cleaned_text'].apply(lambda x: pd.Series(extract_features(x)))

# --- Combine all features ---
# Add original tweet text
feature_df['original_tweet_text'] = df['Tweet Text']
feature_df['cleaned_tweet_text'] = df['cleaned_text']

# Add temporal features from original timestamp
feature_df['hour'] = df['Timestamp'].dt.hour
feature_df['day_of_week'] = df['Timestamp'].dt.dayofweek
feature_df['is_weekend'] = feature_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Add target variables
feature_df['likes'] = df['Likes']
feature_df['retweets'] = df['Retweets']
feature_df['replies'] = df['Replies']

# --- Save processed data ---
feature_df.to_csv('features_datazet.csv', index=False)

print(f"Feature engineering complete! Created {feature_df.shape[1]-3} features.")
print(f"Dataset shape: {feature_df.shape}")