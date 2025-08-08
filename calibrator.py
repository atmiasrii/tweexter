import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class ViralCalibrator:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        self._prepare_data()
    def _prepare_data(self):
        # Clean and transform data
        self.df['log_likes'] = np.log1p(self.df['likes'])
        # Use only features that exist in both the dataset and our standardized MODEL_FEATURES
        from features import MODEL_FEATURES
        available = set(self.df.columns)
        model_features_set = set(MODEL_FEATURES)
        
        # Calibrator feature candidates (subset of viral-relevant features)
        calibrator_candidates = [
            'question_count', 'exclamation_count', 'sentiment_compound',
            'char_count', 'hashtag_count', 'mention_count'
        ]
        
        # Only use features that exist in dataset AND are in our standardized MODEL_FEATURES
        self.features = [
            f for f in calibrator_candidates 
            if f in available and f in model_features_set
        ]
        
        print(f"Calibrator using {len(self.features)} features: {self.features}")
        # Target variables
        self.X = self.scaler.fit_transform(self.df[self.features])
        self.y_reg = self.df['log_likes']
        self.y_clf = self.df['is_viral']
        # Train models
        self.regressor = GradientBoostingRegressor(n_estimators=150, max_depth=5)
        self.classifier = GradientBoostingClassifier(n_estimators=100)
        self.regressor.fit(self.X, self.y_reg)
        self.classifier.fit(self.X, self.y_clf)
    def predict(self, input_features):
        """input_features should be a dict matching self.features"""
        X_input = np.array([[input_features[f] for f in self.features]])
        X_scaled = self.scaler.transform(X_input)
        log_likes = self.regressor.predict(X_scaled)[0]
        viral_prob = self.classifier.predict_proba(X_scaled)[0][1]
        return {
            'predicted_likes': int(np.expm1(log_likes)),
            'virality_prob': float(viral_prob),
            'virality_tier': self._get_tier(viral_prob),
            'key_features': self._get_feature_impacts()
        }
    def _get_tier(self, prob):
        if prob > 0.7: return 'High'
        elif prob > 0.4: return 'Medium'
        return 'Low'
    def _get_feature_impacts(self):
        imp = self.regressor.feature_importances_
        return sorted([
            {'feature': f, 'weight': float(w)} 
            for f, w in zip(self.features, imp)
        ], key=lambda x: -x['weight'])[:3]

# Initialize on import
calibrator = ViralCalibrator('ready_datazet.csv')
