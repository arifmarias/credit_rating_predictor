import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
from ..utils.config import config

logger = logging.getLogger(__name__)

class RatingCalibrator:
    def __init__(self):
        self.industry_adjustments = {}
        self.market_conditions = {}
        self.rating_scale = [
            'AAA', 'AA+', 'AA', 'AA-',
            'A+', 'A', 'A-',
            'BBB+', 'BBB', 'BBB-',
            'BB+', 'BB', 'BB-',
            'B+', 'B', 'B-',
            'CCC+', 'CCC', 'CCC-',
            'CC', 'C', 'D'
        ]
    
    def load_industry_adjustments(self, industry_data: pd.DataFrame):
        """Load industry-specific rating adjustments"""
        self.industry_adjustments = (
            industry_data
            .set_index('industry')
            ['adjustment_factor']
            .to_dict()
        )
    
    def update_market_conditions(self, market_data: Dict[str, float]):
        """Update market condition factors"""
        self.market_conditions = market_data
    
    def _calculate_industry_factor(self, industry: str) -> float:
        """Calculate industry-specific adjustment factor"""
        return self.industry_adjustments.get(industry, 1.0)
    
    def _calculate_market_factor(self) -> float:
        """Calculate market condition adjustment factor"""
        if not self.market_conditions:
            return 1.0
            
        # Weighted average of market conditions
        weights = {
            'market_volatility': 0.3,
            'interest_rates': 0.3,
            'economic_growth': 0.4
        }
        
        market_factor = sum(
            self.market_conditions.get(factor, 0) * weight
            for factor, weight in weights.items()
        )
        
        return market_factor
    
    def _adjust_rating_notches(self, base_rating: str,
                             adjustment: float) -> str:
        """Adjust rating by specified number of notches"""
        try:
            current_index = self.rating_scale.index(base_rating)
            adjusted_index = int(current_index + adjustment)
            adjusted_index = max(0, min(adjusted_index, len(self.rating_scale) - 1))
            return self.rating_scale[adjusted_index]
        except ValueError:
            logger.warning(f"Invalid rating: {base_rating}")
            return base_rating
    
    def calibrate_rating(self, 
                        predicted_rating: str,
                        company_data: Dict[str, Any],
                        sentiment_data: Dict[str, float] = None) -> Dict[str, Any]:
        """Calibrate predicted rating based on various factors"""
        # Calculate adjustment factors
        industry_factor = self._calculate_industry_factor(
            company_data.get('industry', 'unknown')
        )
        market_factor = self._calculate_market_factor()
        
        # Calculate sentiment adjustment if available
        sentiment_adjustment = 0
        if sentiment_data:
            sentiment_score = sentiment_data.get('avg_sentiment', 0)
            sentiment_adjustment = np.sign(sentiment_score) * min(
                abs(sentiment_score), 2
            )
        
        # Combine adjustments
        total_adjustment = (
            industry_factor +
            market_factor +
            sentiment_adjustment
        )
        
        # Apply adjustments
        calibrated_rating = self._adjust_rating_notches(
            predicted_rating,
            total_adjustment
        )
        
        return {
            'original_rating': predicted_rating,
            'calibrated_rating': calibrated_rating,
            'adjustments': {
                'industry': industry_factor,
                'market': market_factor,
                'sentiment': sentiment_adjustment,
                'total': total_adjustment
            }
        }
    
    def batch_calibrate(self, 
                       predictions: pd.DataFrame,
                       companies: pd.DataFrame,
                       sentiments: pd.DataFrame = None) -> pd.DataFrame:
        """Calibrate ratings for multiple predictions"""
        calibrated_ratings = []
        
        for _, row in predictions.iterrows():
            company_data = companies[
                companies['company_id'] == row['company_id']
            ].iloc[0].to_dict()
            
            sentiment_data = None
            if sentiments is not None:
                sentiment_data = sentiments[
                    sentiments['company_id'] == row['company_id']
                ].iloc[0].to_dict()
            
            calibration = self.calibrate_rating(
                row['predicted_rating'],
                company_data,
                sentiment_data
            )
            
            calibrated_ratings.append({
                'company_id': row['company_id'],
                'original_rating': calibration['original_rating'],
                'calibrated_rating': calibration['calibrated_rating'],
                **calibration['adjustments']
            })
        
        return pd.DataFrame(calibrated_ratings)