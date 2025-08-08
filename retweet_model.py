import pandas as pd
import numpy as np
import joblib
import json
import ast
from urllib.parse import urlparse
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Import new modules for topic and trending features  
from topic_category_hf import get_topic_category
from trending_hashtags import count_trending_hashtags

def extract_domain(url):
    """Extract domain from URL string"""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain.lower()
    except:
        return None

def safe_parse_list(value):
    """Safely parse string-represented lists"""
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value) if value and value != "[]" else []
    except (ValueError, SyntaxError):
        return [item.strip() for item in value.split(',')] if value else []

class RetweetPredictor:
    def __init__(self, data_path='ready_datazet.csv'):
        self.df = pd.read_csv(data_path)
        self.trending_data = self._load_trending_data()
        self._clean_data()
        self._prepare_features()
        self.models = []
        self.validation_report = {}
        
    def _clean_data(self):
        """Enhanced cleaning with outlier handling"""
        # Preserve zero-engagement tweets
        self.df['retweets'] = self.df['retweets'].fillna(0)
        
        # Keep zeros but cap extremes
        upper_cap = self.df['retweets'].quantile(0.995)
        self.df['retweets'] = self.df['retweets'].clip(upper=upper_cap)
    
    def _load_trending_data(self):
        """Load trending hashtags data (customize your data source)"""
        # Example structure: {date: set(hashtags)}
        try:
            with open('trending_hashtags.json') as f:
                return {k: set(v) for k, v in json.load(f).items()}
        except:
            print("⚠️ Trending data not found - using empty set")
            return {}
    
    def _count_trending_hashtags(self, hashtags, created_at):
        """Count trending hashtags in tweet (requires trending data)"""
        if not hashtags:
            return 0
        
        # Get trending hashtags for this date (implement your loading logic)
        date_str = str(created_at).split()[0]  # Extract date portion
        trending = self.trending_data.get(date_str, set())
        
        return sum(1 for tag in hashtags if tag.lower() in trending)

    def _check_news_urls(self, urls, news_domains):
        """Check if any URL is from a news domain using proper URL parsing"""
        if not urls:
            return 0
        
        # Parse the URLs list
        url_list = safe_parse_list(urls)
        if not url_list:
            return 0
        
        # Check each URL for news domains
        for url in url_list:
            domain = extract_domain(str(url))
            if domain and domain in news_domains:
                return 1
        return 0

    def _predict_topic(self, text):
        """Categorize tweet topic using HF model with retweet-focused fallback"""
        # Try HF model first
        try:
            return get_topic_category(text, threshold=0.3)
        except Exception:
            # Retweet-focused fallback categories
            text_lower = str(text).lower()
            if 'sports' in text_lower or 'game' in text_lower or 'team' in text_lower:
                return 1  # Sports
            elif 'politics' in text_lower or 'election' in text_lower or 'vote' in text_lower:
                return 2  # Politics
            elif 'tech' in text_lower or 'ai' in text_lower or 'software' in text_lower:
                return 3  # Technology
            elif 'news' in text_lower or 'breaking' in text_lower:
                return 4  # News
            return 0  # General
    
    def _prepare_features(self):
        """Enhanced feature engineering with semantic/NLP features"""
        # Core features
        if 'text' in self.df.columns:
            self.df['text_length'] = self.df['text'].astype(str).apply(len)
        else:
            self.df['text_length'] = self.df.get('char_count', 0)
            
        if 'media_count' in self.df.columns:
            self.df['has_media'] = self.df['media_count'] > 0
        else:
            self.df['has_media'] = 0
            
        if 'created_at' in self.df.columns:
            try:
                self.df['hour'] = pd.to_datetime(self.df['created_at']).dt.hour
            except:
                self.df['hour'] = self.df.get('hour', 12)  # Default to noon
        else:
            self.df['hour'] = self.df.get('hour', 12)
        
        # Sentiment features (use existing or create placeholders)
        sentiment_features = []
        if 'sentiment_compound' in self.df.columns:
            sentiment_features.append('sentiment_compound')
        if 'sentiment_subjectivity' in self.df.columns:
            sentiment_features.append('sentiment_subjectivity')
        
        # Emotion detection features (use existing or create placeholders)
        emotion_features = []
        emotion_cols = ['emotion_anger', 'emotion_fear', 'emotion_joy', 'emotion_sadness', 'emotion_surprise']
        for col in emotion_cols:
            if col in self.df.columns:
                emotion_features.append(col)
            else:
                self.df[col] = 0  # Placeholder
                emotion_features.append(col)
        
        # URL features
        if 'urls' in self.df.columns:
            # Parse URL strings
            self.df['urls'] = self.df['urls'].apply(safe_parse_list)
            
            self.df['has_url'] = self.df['urls'].apply(lambda x: 1 if x else 0)
            
            # News URL detection
            news_domains = {'bbc.com', 'cnn.com', 'nytimes.com', 'washingtonpost.com'}
            self.df['has_news_url'] = self.df['urls'].apply(
                lambda urls: 1 if any(extract_domain(url) in news_domains for url in urls) else 0
            )
        else:
            self.df['has_url'] = 0
            self.df['has_news_url'] = 0
        
        # Trending hashtags - use new module
        if 'text' in self.df.columns:
            # Use text directly with new trending module
            self.df['trending_hashtag_count'] = self.df['text'].apply(
                lambda text: count_trending_hashtags(str(text))
            )
        elif 'hashtags' in self.df.columns:
            # Fallback: parse hashtag strings and use old method
            self.df['hashtags'] = self.df['hashtags'].apply(safe_parse_list)
            self.df['trending_hashtag_count'] = self.df.apply(
                lambda row: self._count_trending_hashtags(row['hashtags'], row['created_at']), 
                axis=1
            )
        else:
            self.df['trending_hashtag_count'] = 0
        
        # Topic/intent category
        if 'text' in self.df.columns:
            self.df['topic_category'] = self.df['text'].apply(self._predict_topic)
        else:
            self.df['topic_category'] = 0
        
        # Enhanced feature selection combining existing and new features
        base_features = [
            'char_count', 'word_count', 'sentence_count',
            'hashtag_count', 'mention_count', 'question_count', 
            'exclamation_count', 'day_of_week', 'is_weekend'
        ]
        
        # Filter base features that exist in dataset
        existing_base_features = [f for f in base_features if f in self.df.columns]
        
        # New semantic/NLP features
        new_features = [
            'text_length', 'has_media', 'hour', 'has_url', 
            'has_news_url', 'trending_hashtag_count', 'topic_category'
        ]
        
        # Combine all features
        self.features = existing_base_features + new_features + sentiment_features + emotion_features
        
        # Handle missing values
        self.X = self.df[self.features].fillna(0)
        self.y = np.log1p(self.df['retweets'])  # Log transform for better distribution
        
    def train_ensemble(self):
        """Train new ensemble models with different seeds"""
        seeds = [42, 99, 123, 456, 789]
        self.models = []
        
        print("Training ensemble models...")
        for seed in seeds:
            print(f"Training model with seed {seed}...")
            model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=seed,
                loss='huber'  # Robust to outliers
            )
            model.fit(self.X, self.y)
            self.models.append(model)
            joblib.dump(model, f'NEW_retweet_model_seed_{seed}.pkl')
        
        print(f"✅ Trained {len(self.models)} ensemble models")
    
    def train_quantile(self):
        """Train new quantile model for robust median prediction"""
        print("Training quantile model...")
        self.quantile_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            alpha=0.5,  # For median
            loss='quantile',
            random_state=42
        )
        self.quantile_model.fit(self.X, self.y)
        joblib.dump(self.quantile_model, 'NEW_retweet_quantile_model.pkl')
        print("✅ Trained quantile model")
    
    def evaluate_models(self):
        """Evaluate new models on test set"""
        # Create test set
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        results = {}
        
        # Evaluate ensemble models
        if self.models:
            ensemble_preds = []
            for model in self.models:
                pred = model.predict(X_test)
                ensemble_preds.append(pred)
            
            mean_pred = np.mean(ensemble_preds, axis=0)
            mean_rt = np.expm1(mean_pred)
            actual_rt = np.expm1(y_test)
            
            results['ensemble'] = {
                'mae': mean_absolute_error(actual_rt, mean_rt),
                'rmse': np.sqrt(mean_squared_error(actual_rt, mean_rt))
            }
        
        # Evaluate quantile model
        if hasattr(self, 'quantile_model'):
            quantile_pred = self.quantile_model.predict(X_test)
            quantile_rt = np.expm1(quantile_pred)
            actual_rt = np.expm1(y_test)
            
            results['quantile'] = {
                'mae': mean_absolute_error(actual_rt, quantile_rt),
                'rmse': np.sqrt(mean_squared_error(actual_rt, quantile_rt))
            }
            
            # Save test predictions
            test_results = pd.DataFrame({
                'actual': actual_rt,
                'ensemble_pred': mean_rt if self.models else quantile_rt,
                'quantile_pred': quantile_rt
            })
            test_results.to_csv('NEW_retweet_test_predictions.csv', index=False)
        
        return results
    
    def predict(self, tweet_features, method='quantile'):
        """Predict with new models and safety caps"""
        # Validate input features
        missing = [f for f in self.features if f not in tweet_features]
        if missing:
            print(f"⚠️ Missing features: {missing}. Using defaults.")
            for f in missing:
                tweet_features[f] = 0  # Or appropriate default
        
        # Prepare input
        input_df = pd.DataFrame([tweet_features])
        input_df = input_df[self.features].fillna(0)
        
        # Select prediction method
        if method == 'ensemble' and self.models:
            # Ensemble prediction
            preds = [model.predict(input_df)[0] for model in self.models]
            log_pred = np.mean(preds)
        elif hasattr(self, 'quantile_model'):
            # Quantile prediction
            log_pred = self.quantile_model.predict(input_df)[0]
        else:
            raise ValueError("No trained models available for prediction")
            
        # Convert to original scale
        prediction = np.expm1(log_pred)
        
        # Apply caps
        capped_prediction = min(prediction, 100000)
        
        # Return both raw and capped predictions
        return {
            'method': method,
            'raw_prediction': prediction,
            'capped_prediction': capped_prediction,
            'was_capped': prediction != capped_prediction
        }

    def generate_report(self):
        """Generate comprehensive validation report"""
        # Evaluate models
        self.validation_report = self.evaluate_models()
        
        # Add feature importance
        if self.models:
            importances = self.models[0].feature_importances_
            self.validation_report['feature_importances'] = dict(zip(self.features, importances))
        
        # Add model metadata
        self.validation_report['models_trained'] = {
            'ensemble_models': len(self.models) if hasattr(self, 'models') else 0,
            'quantile_model': hasattr(self, 'quantile_model')
        }
        
        # Save report
        with open('NEW_retweet_validation_report.json', 'w') as f:
            json.dump(self.validation_report, f, indent=2)
            
        return self.validation_report

if __name__ == "__main__":
    print("Building new retweet prediction models...")
    predictor = RetweetPredictor()
    
    # Train new models
    predictor.train_ensemble()
    predictor.train_quantile()
    
    # Generate and print report
    report = predictor.generate_report()
    print("\n=== Validation Report ===")
    print(f"Ensemble Model MAE: {report.get('ensemble', {}).get('mae', 'N/A'):.2f}")
    print(f"Quantile Model MAE: {report.get('quantile', {}).get('mae', 'N/A'):.2f}")
    
    # Test prediction
    test_features = {f: 0 for f in predictor.features}  # Default test
    test_features['char_count'] = 100
    test_features['sentiment_compound'] = 0.5
    
    print("\nTest Prediction:")
    print("Quantile:", predictor.predict(test_features, method='quantile'))
    print("Ensemble:", predictor.predict(test_features, method='ensemble'))
    
    print("\n✅ New model training complete")
