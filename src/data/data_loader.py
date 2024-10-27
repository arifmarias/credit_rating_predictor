from pathlib import Path
import pandas as pd
from typing import Tuple, Dict
from datetime import datetime
import logging
from ..utils.config import config

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        self.config = config.data
        
    def load_news_articles(self) -> pd.DataFrame:
        """Load news articles dataset"""
        df = pd.read_csv(self.config.news_file)
        required_cols = ['article_id', 'company_id', 'text', 'date']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"News articles file missing required columns: {required_cols}")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def load_company_profiles(self) -> pd.DataFrame:
        """Load company profiles"""
        df = pd.read_csv(self.config.company_file)
        required_cols = ['company_id', 'industry', 'size', 'country']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Company profiles file missing required columns: {required_cols}")
        return df
    
    def load_credit_ratings(self) -> pd.DataFrame:
        """Load historical credit ratings"""
        df = pd.read_csv(self.config.ratings_file)
        required_cols = ['company_id', 'rating', 'date']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Credit ratings file missing required columns: {required_cols}")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def load_financial_metrics(self) -> pd.DataFrame:
        """Load financial metrics"""
        df = pd.read_csv(self.config.metrics_file)
        required_cols = ['company_id', 'date', 'revenue', 'debt_ratio', 'current_ratio']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Financial metrics file missing required columns: {required_cols}")
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def combine_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Combine all data sources into features and targets"""
        news_df = self.load_news_articles()
        companies_df = self.load_company_profiles()
        ratings_df = self.load_credit_ratings()
        metrics_df = self.load_financial_metrics()
        
        # Group news articles by company and calculate aggregates
        news_features = (
            news_df.groupby('company_id')
            .agg({
                'text': list,
                'date': lambda x: list(x)
            })
            .reset_index()
        )
        
        # Get latest ratings for each company
        latest_ratings = (
            ratings_df
            .sort_values('date')
            .groupby('company_id')
            .last()
            .reset_index()
        )
        
        # Combine all features
        combined_df = (
            companies_df
            .merge(news_features, on='company_id', how='left')
            .merge(latest_ratings[['company_id', 'rating']], on='company_id', how='left')
            .merge(metrics_df, on='company_id', how='left')
        )
        
        return combined_df

    def get_train_test_split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        # Shuffle data
        df = df.sample(frac=1, random_state=config.random_seed)
        
        # Calculate split indices
        train_idx = int(len(df) * self.config.train_ratio)
        val_idx = int(len(df) * (self.config.train_ratio + self.config.validation_ratio))
        
        return {
            'train': df[:train_idx],
            'validation': df[train_idx:val_idx],
            'test': df[val_idx:]
        }