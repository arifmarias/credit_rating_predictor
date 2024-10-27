from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Any
import yaml

class ModelConfig(BaseModel):
    """Model configuration settings"""
    topic_model_name: str = "llama2"
    sentiment_model_name: str = "ProsusAI/finbert"
    rating_model_type: str = "random_forest"
    use_gpu: bool = True
    batch_size: int = 32
    max_length: int = 512

class DataConfig(BaseModel):
    """Data configuration settings"""
    news_file: Path = Path("data/news_articles.csv")
    company_file: Path = Path("data/company_profiles.csv")
    ratings_file: Path = Path("data/credit_ratings.csv")
    metrics_file: Path = Path("data/financial_metrics.csv")
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1

class Config(BaseModel):
    """Main configuration"""
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    random_seed: int = 42
    log_level: str = "INFO"
    output_dir: Path = Path("output")

    @classmethod
    def from_yaml(cls, yaml_file: Path) -> "Config":
        """Load configuration from YAML file"""
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

config = Config.from_yaml(Path("config.yaml"))