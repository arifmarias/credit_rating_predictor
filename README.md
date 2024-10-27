# Credit Rating Prediction System

A comprehensive system for predicting credit ratings using large language models (LLama2), financial metrics, and news sentiment analysis.

## Project Structure
```
credit_rating_predictor/
├── src/
│   ├── data_generation/        # Data generation using LLama2
│   │   ├── __init__.py
│   │   └── llama_generator.py
│   ├── models/                 # Model implementations
│   │   ├── topic_classifier.py
│   │   ├── sentiment_analyzer.py
│   │   └── rating_predictor.py
│   ├── data/                   # Data processing modules
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── utils/                  # Utility functions
│   │   └── config.py
│   └── calibration/            # Rating calibration
│       └── calibrator.py
├── scripts/                    # Utility scripts
│   └── generate_data.py
├── data/                       # Data storage
├── logs/                       # Log files
├── requirements.txt
├── config.yaml
└── main.py
```

## Detailed Component Description

### 1. Configuration System

#### config.yaml
**What it does:**
- Stores external configuration settings
- Defines model parameters
- Sets data paths and processing parameters

**Why we need it:**
- Allows parameter changes without code modification
- Supports different environments (dev, prod, test)
- Makes configuration transparent and maintainable

**How it works:**
```yaml
model:
  topic_model_name: "llama2"
  sentiment_model_name: "ProsusAI/finbert"
  rating_model_type: "random_forest"
  use_gpu: true
  batch_size: 32
  max_length: 512

data:
  news_file: "data/news_articles.csv"
  company_file: "data/company_profiles.csv"
```

#### config.py
**What it does:**
- Validates configuration settings
- Provides type safety
- Sets default values
- Makes configuration accessible throughout the code

**Why we need it:**
- Ensures configuration correctness
- Provides IDE support
- Centralizes configuration management

**How it works:**
```python
class ModelConfig(BaseModel):
    topic_model_name: str = "llama2"
    sentiment_model_name: str = "ProsusAI/finbert"
    rating_model_type: str = "random_forest"
    use_gpu: bool = True
    batch_size: int = 32
```

### 2. Data Generation System (llama_generator.py)

**What it does:**
- Generates synthetic financial data using LLama2
- Creates realistic company profiles
- Generates news articles
- Produces financial metrics
- Creates credit rating histories

**Why we need it:**
- Provides controlled testing data
- Enables development without real data
- Ensures consistent data format

**How it works:**
```python
# Initialize generator
generator = LlamaDataGenerator(
    n_companies=100,
    n_articles_per_company=5
)

# Generate data
datasets = generator.generate_all_data()
```

### 3. Data Processing System

#### data_loader.py
**What it does:**
- Loads data from various sources
- Validates data structure
- Combines different data types

**Why we need it:**
- Centralizes data loading logic
- Ensures data consistency
- Handles data validation

**How it works:**
```python
loader = DataLoader()
news_data = loader.load_news_articles()
company_data = loader.load_company_profiles()
```

#### preprocessor.py
**What it does:**
- Cleans and prepares data
- Creates features for models
- Handles different data types
- Normalizes numerical data

**Why we need it:**
- Prepares data for modeling
- Ensures consistent feature format
- Improves model performance

**How it works:**
```python
preprocessor = DataPreprocessor()
X, feature_names = preprocessor.prepare_features(
    combined_data,
    ratings_data
)
```

### 4. Model System

#### topic_classifier.py
**What it does:**
- Classifies news articles into categories
- Uses LLama2 for classification
- Provides confidence scores

**Why we need it:**
- Organizes news by topic
- Enables topic-based analysis
- Provides feature for rating prediction

#### sentiment_analyzer.py
**What it does:**
- Analyzes news sentiment
- Uses FinBERT model
- Provides sentiment scores

**Why we need it:**
- Captures market sentiment
- Adds qualitative factors
- Enhances rating prediction

#### rating_predictor.py
**What it does:**
- Predicts credit ratings
- Combines multiple data sources
- Provides confidence scores

**Why we need it:**
- Core rating prediction
- Integrates various signals
- Produces initial ratings

## Execution Sequence

![alt text](Dataflow.png)