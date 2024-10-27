import logging
import time
from typing import List, Dict, Any, Optional
import requests
from pydantic import BaseModel
from enum import Enum
import pandas as pd
from ..utils.config import config

logger = logging.getLogger(__name__)

class TopicCategory(str, Enum):
    EARNINGS = "earnings"
    MARKET_PERFORMANCE = "market_performance"
    MERGER_ACQUISITION = "merger_acquisition"
    REGULATORY = "regulatory"
    PRODUCT_LAUNCH = "product_launch"
    LEADERSHIP_CHANGE = "leadership_change"
    INDUSTRY_TRENDS = "industry_trends"
    FINANCIAL_RESULTS = "financial_results"
    STRATEGIC_INITIATIVE = "strategic_initiative"
    MARKET_EXPANSION = "market_expansion"
    PARTNERSHIP = "partnership"
    INVESTMENT = "investment"
    RESTRUCTURING = "restructuring"
    OTHERS = "others"

class TopicClassification(BaseModel):
    article_id: str
    primary_topic: TopicCategory
    confidence: float
    secondary_topics: Optional[List[TopicCategory]] = None
    keywords: Optional[List[str]] = None

class TopicClassifier:
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self._verify_connection()
        
    def _verify_connection(self):
        """Verify Ollama connection"""
        try:
            response = requests.get(
                self.api_url.replace("/generate", "/version"),
                timeout=5
            )
            response.raise_for_status()
            logger.info("Successfully connected to Ollama")
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Ollama. Please ensure it's running: {str(e)}"
            )

    def _generate_prompt(self, text: str) -> str:
        """Generate classification prompt"""
        categories = [
            f"{i+1}. {cat.value}: {self._get_category_description(cat)}"
            for i, cat in enumerate(TopicCategory)
        ]
        categories_text = "\n".join(categories)
        
        return f"""Classify this financial news article into one primary topic category.

Available categories:
{categories_text}

Rules:
1. Choose ONE primary category number (1-{len(TopicCategory)})
2. Identify up to 3 relevant keywords
3. If applicable, suggest ONE secondary category
4. Assign a confidence score (0.0 to 1.0)

Return the response in this exact format:
PRIMARY: [category number]
CONFIDENCE: [score]
SECONDARY: [category number or 'none']
KEYWORDS: [comma-separated keywords]

Article text:
{text}

Classification:"""

    def _get_category_description(self, category: TopicCategory) -> str:
        """Get description for each category"""
        descriptions = {
            TopicCategory.EARNINGS: "Quarterly/annual earnings reports, financial performance",
            TopicCategory.MARKET_PERFORMANCE: "Stock price movements, market valuation",
            TopicCategory.MERGER_ACQUISITION: "M&A activities, takeovers, acquisitions",
            TopicCategory.REGULATORY: "Compliance, regulatory changes, legal matters",
            TopicCategory.PRODUCT_LAUNCH: "New product announcements, launches",
            TopicCategory.LEADERSHIP_CHANGE: "Executive appointments, management changes",
            TopicCategory.INDUSTRY_TRENDS: "Market trends, industry analysis",
            TopicCategory.FINANCIAL_RESULTS: "Financial statements, performance metrics",
            TopicCategory.STRATEGIC_INITIATIVE: "Strategic plans, corporate initiatives",
            TopicCategory.MARKET_EXPANSION: "Geographic expansion, new market entry",
            TopicCategory.PARTNERSHIP: "Strategic partnerships, collaborations",
            TopicCategory.INVESTMENT: "Investments, funding, capital allocation",
            TopicCategory.RESTRUCTURING: "Organization changes, restructuring",
            TopicCategory.OTHERS: "Topics not covered by other categories"
        }
        return descriptions.get(category, "")

    def _call_llm(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Call Ollama API with retry logic"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.3  # Lower temperature for more consistent classification
                    },
                    timeout=30
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response"""
        try:
            result = {
                'primary_topic': None,
                'confidence': 0.0,
                'secondary_topics': None,
                'keywords': []
            }
            
            for line in response.split('\n'):
                line = line.strip()
                if line.startswith('PRIMARY:'):
                    category_num = int(line.replace('PRIMARY:', '').strip()) - 1
                    result['primary_topic'] = list(TopicCategory)[category_num].value
                elif line.startswith('CONFIDENCE:'):
                    result['confidence'] = float(line.replace('CONFIDENCE:', '').strip())
                elif line.startswith('SECONDARY:'):
                    secondary = line.replace('SECONDARY:', '').strip()
                    if secondary.lower() != 'none':
                        category_num = int(secondary) - 1
                        result['secondary_topics'] = [list(TopicCategory)[category_num].value]
                elif line.startswith('KEYWORDS:'):
                    keywords = line.replace('KEYWORDS:', '').strip()
                    result['keywords'] = [k.strip() for k in keywords.split(',')]
            
            return result
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return {
                'primary_topic': TopicCategory.OTHERS.value,
                'confidence': 0.5,
                'secondary_topics': None,
                'keywords': []
            }

    def classify_article(self, article_id: str, text: str) -> TopicClassification:
        """Classify a single article"""
        try:
            prompt = self._generate_prompt(text)
            response = self._call_llm(prompt)
            result = self._parse_response(response['response'])
            
            return TopicClassification(
                article_id=article_id,
                **result
            )
        except Exception as e:
            logger.error(f"Error classifying article {article_id}: {str(e)}")
            return TopicClassification(
                article_id=article_id,
                primary_topic=TopicCategory.OTHERS,
                confidence=0.0
            )

    def classify_batch(self, articles: pd.DataFrame, 
                      text_column: str = 'text',
                      batch_size: int = 10) -> List[TopicClassification]:
        """Classify a batch of articles"""
        results = []
        total = len(articles)
        
        for i in range(0, total, batch_size):
            batch = articles.iloc[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total-1)//batch_size + 1}")
            
            for _, row in batch.iterrows():
                result = self.classify_article(
                    str(row['article_id']),
                    row[text_column]
                )
                results.append(result)
                time.sleep(0.5)  # Prevent overwhelming the API
        
        return results

    def get_topic_distribution(self, classifications: List[TopicClassification]) -> Dict[str, float]:
        """Get distribution of topics across classifications"""
        total = len(classifications)
        topic_counts = {}
        
        for classification in classifications:
            topic = classification.primary_topic
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return {
            topic: count/total 
            for topic, count in topic_counts.items()
        }