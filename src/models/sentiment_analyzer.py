from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Any
import logging
from ..utils.config import config

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.model.use_gpu else 'cpu')
        self.model_name = config.model.sentiment_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        
    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Analyze sentiment of texts"""
        results = []
        
        for text in texts:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config.model.max_length,
                padding=True
            ).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(scores, dim=1)
            
            # Convert to sentiment labels
            sentiment_label = ['negative', 'neutral', 'positive'][prediction.item()]
            confidence = scores[0][prediction].item()
            
            results.append({
                'sentiment': sentiment_label,
                'confidence': confidence,
                'scores': {
                    'negative': scores[0][0].item(),
                    'neutral': scores[0][1].item(),
                    'positive': scores[0][2].item()
                }
            })
        
        return results
    
    def get_sentiment_features(self, texts: List[str]) -> Dict[str, float]:
        """Extract sentiment features from texts"""
        sentiments = self.analyze_sentiment(texts)
        
        # Calculate aggregate features
        sentiment_scores = [1 if s['sentiment'] == 'positive' else (-1 if s['sentiment'] == 'negative' else 0)
                          for s in sentiments]
        
        return {
            'avg_sentiment': np.mean(sentiment_scores),
            'sentiment_std': np.std(sentiment_scores),
            'positive_ratio': sum(1 for s in sentiments if s['sentiment'] == 'positive') / len(sentiments),
            'negative_ratio': sum(1 for s in sentiments if s['sentiment'] == 'negative') / len(sentiments),
            'avg_confidence': np.mean([s['confidence'] for s in sentiments])
        }