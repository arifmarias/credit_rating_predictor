import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from datetime import datetime
from ..utils.config import config

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            stop_words='english'
        )
        self.feature_columns = []
        
    def preprocess_text_features(self, texts: List[str]) -> np.ndarray:
        """Preprocess text data and convert to TF-IDF features"""
        # Basic text cleaning
        processed_texts = []
        for text in texts:
            if isinstance(text, list):
                # If we have multiple texts per company, concatenate them
                text = ' '.join(text)
            text = str(text).lower()
            text = ' '.join(text.split())  # Normalize whitespace
            processed_texts.append(text)
        
        # Convert to TF-IDF features
        tfidf_features = self.tfidf.fit_transform(processed_texts)
        return tfidf_features.toarray()
    
    def preprocess_financial_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess financial metrics"""
        metrics_cols = [
            'revenue', 'debt_ratio', 'current_ratio', 
            'quick_ratio', 'roa', 'roe', 'operating_margin'
        ]
        
        # Create derived metrics
        df['debt_to_equity'] = df['debt_ratio'] / (1 - df['debt_ratio'])
        df['interest_coverage'] = df['operating_margin'] / df['debt_ratio']
        
        # Handle missing values with forward fill and mean
        for col in metrics_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
                df[col] = df[col].fillna(df[col].mean())
        
        # Handle infinities
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())
        
        # Scale numerical features
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        categorical_cols = ['industry', 'size', 'country', 'region']
        
        df_encoded = df.copy()
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                
                # Handle missing values
                df_encoded[col] = df_encoded[col].fillna('UNKNOWN')
                
                # Fit and transform
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col])
        
        return df_encoded
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from dates"""
        if 'date' not in df.columns:
            return df
            
        df = df.copy()
        
        # Basic date features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['dayofweek'] = df['date'].dt.dayofweek
        
        # Calculate time-based statistics
        df['days_since_last_update'] = (
            df.groupby('company_id')['date']
            .diff()
            .dt.days
        )
        
        # Create cyclical features for month and dayofweek
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['day_sin'] = np.sin(2 * np.pi * df['dayofweek']/7)
        df['day_cos'] = np.cos(2 * np.pi * df['dayofweek']/7)
        
        return df
    
    def create_rating_history_features(self, df: pd.DataFrame, 
                                     ratings_history: pd.DataFrame) -> pd.DataFrame:
        """Create features from rating history"""
        df = df.copy()
        
        # Calculate rating stability metrics
        rating_changes = (
            ratings_history
            .groupby('company_id')
            .agg({
                'rating': ['count', 'nunique'],
                'date': ['min', 'max']
            })
        )
        
        rating_changes.columns = [
            'rating_updates',
            'unique_ratings',
            'first_rating_date',
            'last_rating_date'
        ]
        
        # Calculate rating history duration
        rating_changes['rating_history_days'] = (
            rating_changes['last_rating_date'] - 
            rating_changes['first_rating_date']
        ).dt.days
        
        # Calculate change frequency
        rating_changes['rating_change_frequency'] = (
            rating_changes['rating_updates'] / 
            rating_changes['rating_history_days']
        )
        
        df = df.merge(rating_changes, on='company_id', how='left')
        
        return df
    
    def prepare_features(self, 
                        df: pd.DataFrame, 
                        ratings_history: pd.DataFrame = None) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare all features for model input"""
        processed_df = df.copy()
        
        # Process text features if present
        if 'text' in df.columns:
            text_features = self.preprocess_text_features(df['text'])
            text_feature_names = [f'text_feature_{i}' for i in range(text_features.shape[1])]
            text_df = pd.DataFrame(
                text_features,
                columns=text_feature_names,
                index=df.index
            )
            processed_df = pd.concat([processed_df, text_df], axis=1)
        
        # Process financial metrics
        processed_df = self.preprocess_financial_metrics(processed_df)
        
        # Encode categorical features
        processed_df = self.encode_categorical_features(processed_df)
        
        # Create temporal features
        processed_df = self.create_temporal_features(processed_df)
        
        # Create rating history features if available
        if ratings_history is not None:
            processed_df = self.create_rating_history_features(
                processed_df, 
                ratings_history
            )
        
        # Store and return feature columns
        self.feature_columns = [col for col in processed_df.columns 
                              if col not in ['company_id', 'text', 'date', 'rating']]
        
        return processed_df[self.feature_columns], self.feature_columns
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessor"""
        if not self.feature_columns:
            raise ValueError("Preprocessor has not been fitted. Run prepare_features first.")
        
        processed_df = df.copy()
        
        # Apply same transformations as in prepare_features
        if 'text' in df.columns and hasattr(self, 'tfidf'):
            text_features = self.tfidf.transform(df['text']).toarray()
            text_feature_names = [f'text_feature_{i}' for i in range(text_features.shape[1])]
            text_df = pd.DataFrame(
                text_features,
                columns=text_feature_names,
                index=df.index
            )
            processed_df = pd.concat([processed_df, text_df], axis=1)
        
        # Transform numerical features
        if hasattr(self, 'scaler'):
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            processed_df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        # Transform categorical features
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                processed_df[col] = processed_df[col].fillna('UNKNOWN')
                processed_df[col] = encoder.transform(processed_df[col])
        
        return processed_df[self.feature_columns]