from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import logging
import joblib
from pathlib import Path
from ..utils.config import config

logger = logging.getLogger(__name__)

class RatingPredictor:
    def __init__(self):
        self.model = None
        self.feature_importance = None
        self.feature_names = None
        self.model_type = config.model.rating_model_type
        
    def _create_model(self, trial: optuna.Trial = None) -> Any:
        """Create the prediction model with optional hyperparameter optimization"""
        if self.model_type == "random_forest":
            if trial is None:
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=config.random_seed
                )
            else:
                return RandomForestClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    max_depth=trial.suggest_int('max_depth', 3, 30),
                    min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
                    min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
                    random_state=config.random_seed
                )
        elif self.model_type == "xgboost":
            if trial is None:
                return xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=config.random_seed
                )
            else:
                return xgb.XGBClassifier(
                    n_estimators=trial.suggest_int('n_estimators', 50, 300),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3),
                    max_depth=trial.suggest_int('max_depth', 3, 15),
                    subsample=trial.suggest_float('subsample', 0.5, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    random_state=config.random_seed
                )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series,
                               n_trials: int = 100) -> Dict[str, Any]:
        """Optimize model hyperparameters using Optuna"""
        def objective(trial):
            model = self._create_model(trial)
            model.fit(X_train, y_train)
            return model.score(X_val, y_val)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best trial: {study.best_trial.params}")
        return study.best_trial.params
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None,
              optimize: bool = True) -> Dict[str, float]:
        """Train the rating prediction model"""
        self.feature_names = X_train.columns.tolist()
        
        if optimize and X_val is not None and y_val is not None:
            best_params = self.optimize_hyperparameters(
                X_train, y_train, X_val, y_val
            )
            self.model = self._create_model()
            for param, value in best_params.items():
                setattr(self.model, param, value)
        else:
            self.model = self._create_model()
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        
        # Calculate metrics
        metrics = {}
        metrics['train_score'] = self.model.score(X_train, y_train)
        if X_val is not None and y_val is not None:
            metrics['val_score'] = self.model.score(X_val, y_val)
            y_pred = self.model.predict(X_val)
            metrics['classification_report'] = classification_report(y_val, y_pred)
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ratings for new data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict rating probabilities for new data"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError("Probability prediction not supported for this model")
    
    def get_feature_importance(self, top_n: int = None) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.feature_importance is None:
            raise ValueError("Feature importance not available")
        
        sorted_importance = dict(sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        if top_n is not None:
            return dict(list(sorted_importance.items())[:top_n])
        return sorted_importance

    def save_model(self, path: Path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_data = {
            'model': self.model,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Path):
        """Load a trained model"""
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.feature_importance = model_data['feature_importance']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        
        logger.info(f"Model loaded from {path}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the model"""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return {
            'model_type': self.model_type,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'model_params': self.model.get_params(),
            'top_features': list(self.get_feature_importance(10).items())
            if self.feature_importance else None
        }