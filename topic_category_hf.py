# topic_category_hf.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Labels from the model card (order matters)
TOPIC_LABELS = [
    'arts_&_culture','business_&_entrepreneurs','celebrity_&_pop_culture',
    'diaries_&_daily_life','family','fashion_&_style','film_tv_&_video',
    'fitness_&_health','food_&_dining','gaming','learning_&_educational',
    'music','news_&_social_concern','other_hobbies','relationships',
    'science_&_technology','sports','travel_&_adventure','youth_&_student_life'
]

_MODEL_NAME = "cardiffnlp/twitter-roberta-base-dec2021-tweet-topic-multi-all"
_tokenizer = None
_model = None

def _lazy_load():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        _model.eval()

def get_topic_category(text: str, threshold: float = 0.3) -> int:
    """
    Returns an integer category id (0..len(TOPIC_LABELS)) where 0 = 'unknown/other'.
    Picks the highest-prob topic if any prob >= threshold; else returns 0.
    """
    try:
        _lazy_load()
        with torch.no_grad():
            inputs = _tokenizer(text, truncation=True, return_tensors="pt")
            logits = _model(**inputs).logits
            probs = torch.sigmoid(logits).squeeze(0).tolist()

        best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        best_prob = probs[best_idx]
        if best_prob >= threshold:
            # Shift by +1 so 0 means "unknown", 1..N are real topics
            return best_idx + 1
        return 0
    except Exception:
        # Fallback to basic heuristics if model fails
        text_lower = text.lower()
        if any(w in text_lower for w in ["sports", "game", "team", "football", "basketball"]):
            return 17  # sports
        elif any(w in text_lower for w in ["politics", "election", "vote", "government"]):
            return 12  # news_&_social_concern
        elif any(w in text_lower for w in ["tech", "ai", "software", "coding"]):
            return 16  # science_&_technology
        elif any(w in text_lower for w in ["music", "song", "album", "artist"]):
            return 12  # music
        else:
            return 0  # unknown/other
