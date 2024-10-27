import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

from src.data.data_loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.rating_predictor import RatingPredictor
from src.calibration.calibrator import RatingCalibrator
from src.utils.config import config

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"credit_rating_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting credit rating prediction process")
    
    try:
        # Initialize components
        data_loader = DataLoader()
        preprocessor = DataPreprocessor()
        sentiment_analyzer = SentimentAnalyzer()
        rating_predictor = RatingPredictor()
        calibrator = RatingCalibrator()
        
        # Load data
        logger.info("Loading data...")
        news_data = data_loader.load_news_articles()
        company_data = data_loader.load_company_profiles()
        ratings_data = data_loader.load_credit_ratings()
        financial_data = data_loader.load_financial_metrics()
        
        # Preprocess data
        logger.info("Preprocessing data...")
        combined_data = data_loader.combine_data()
        X, feature_names = preprocessor.prepare_features(
            combined_data,
            ratings_data
        )
        y = combined_data['rating']
        
        # Split data
        logger.info("Splitting data...")
        data_splits = data_loader.get_train_test_split(
            pd.concat([X, y], axis=1)
        )
        
        # Train model
        logger.info("Training model...")
        metrics = rating_predictor.train(
            data_splits['train'][feature_names],
            data_splits['train']['rating'],
            data_splits['validation'][feature_names],
            data_splits['validation']['rating'],
            optimize=True
        )
        
        logger.info(f"Training metrics: {metrics}")
        
        # Make predictions on test set
        logger.info("Making predictions...")
        test_predictions = rating_predictor.predict(
            data_splits['test'][feature_names]
        )
        
        # Analyze sentiments for test set
        logger.info("Analyzing sentiments...")
        test_sentiments = sentiment_analyzer.analyze_sentiment(
            data_splits['test']['text'].tolist()
        )
        
        # Calibrate ratings
        logger.info("Calibrating ratings...")
        predictions_df = pd.DataFrame({
            'company_id': data_splits['test']['company_id'],
            'predicted_rating': test_predictions
        })
        
        calibrated_ratings = calibrator.batch_calibrate(
            predictions_df,
            company_data,
            pd.DataFrame(test_sentiments)
        )
        
        # Save results
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        calibrated_ratings.to_csv(
            output_dir / f"ratings_{timestamp}.csv",
            index=False
        )
        
        # Save model
        rating_predictor.save_model(
            output_dir / f"model_{timestamp}.joblib"
        )
        
        logger.info("Credit rating prediction process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in credit rating prediction process: {str(e)}")
        raise

if __name__ == "__main__":
    main()