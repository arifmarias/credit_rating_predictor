import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import requests
import time
import logging
from typing import List, Dict, Any
import os
from tqdm import tqdm
import torch

# Enhanced logging setup with colors (using ANSI escape codes)
class ColorFormatter(logging.Formatter):
    """Custom formatter for colored log output"""
    COLORS = {
        'INFO': '\033[92m',  # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',  # Red
        'DEBUG': '\033[94m',  # Blue
        'CRITICAL': '\033[95m',  # Purple
        'RESET': '\033[0m'  # Reset color
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        record.msg = f"{color}{record.msg}{reset}"
        return super().format(record)

# Setup logging with custom formatter
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

class LlamaDataGenerator:
    def __init__(self, 
                 n_companies=10, 
                 n_articles_per_company=5,
                 model_name="llama2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.model.use_gpu else 'cpu')
        self.n_companies = n_companies
        self.n_articles_per_company = n_articles_per_company
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        
        # Constants for data generation
        self.ratings = [
            'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-',
            'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-',
            'B+', 'B', 'B-'
        ]
        self.outlooks = ['stable', 'positive', 'negative', 'watch']
        
        # Generate company IDs
        self.company_ids = [
            f'COMP{str(i).zfill(4)}' 
            for i in range(1, n_companies + 1)
        ]
        
        # Verify Llama connection
        self._verify_llama_connection()

    def _verify_llama_connection(self):
        """Verify that Ollama/Llama2 is running"""
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

    def _generate_llm_response(self, prompt: str, 
                             max_retries: int = 3,
                             delay: int = 1) -> str:
        """Generate response from Llama2 with retry logic"""
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.7
                    },
                    timeout=60
                )
                response.raise_for_status()
                return response.json()["response"].strip()
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    raise Exception(f"Failed to generate response after {max_retries} attempts")
    
    def generate_company_profiles(self) -> pd.DataFrame:
        """Generate company profiles using Llama2"""
        logger.info(f"Starting company profile generation for {self.n_companies} companies...")
        data = []
        
        company_prompt = """Generate a realistic company profile in JSON format with the following fields:
        - industry (choose from: technology, banking, healthcare, retail, manufacturing, energy)
        - size (choose from: small, medium, large)
        - country (choose a realistic country)
        - region (choose appropriate region based on country)
        - main_product (main product or service)
        - year_founded (realistic founding year)
        - sector (specific sector within industry)
        - employee_count (realistic number based on size)
        - market_cap_category (small-cap, mid-cap, large-cap)
        
        Return ONLY the JSON, no other text."""
        
        # Using tqdm for progress bar
        for comp_id in tqdm(self.company_ids, desc="Generating company profiles"):
            try:
                response = self._generate_llm_response(company_prompt)
                # Clean and parse the JSON response
                json_str = response.strip().replace("```json", "").replace("```", "")
                company_data = eval(json_str)  # Convert string to dict
                company_data['company_id'] = comp_id
                data.append(company_data)
                
                logger.debug(f"Generated profile for {comp_id}")
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate profile for {comp_id}: {str(e)}")
                # Add fallback data if generation fails
                data.append({
                    'company_id': comp_id,
                    'industry': random.choice(['technology', 'banking', 'healthcare']),
                    'size': random.choice(['small', 'medium', 'large']),
                    'country': 'USA',
                    'region': 'North America',
                    'main_product': 'General Services',
                    'year_founded': random.randint(1950, 2020),
                    'sector': 'General',
                    'employee_count': random.randint(100, 10000),
                    'market_cap_category': 'mid-cap'
                })
        
        df = pd.DataFrame(data)
        logger.info(f"✅ Successfully generated {len(data)} company profiles")
        return df
    
    def generate_news_articles(self) -> pd.DataFrame:
        """Generate news articles using Llama2"""
        total_articles = self.n_companies * self.n_articles_per_company
        logger.info(f"Starting news article generation ({total_articles} articles)...")
        data = []
        article_id = 1
        
        article_prompt = """Generate a financial news article about the company {company_id} with the following specifications:
        - Write a headline (one line)
        - Write a news article (2-3 paragraphs)
        - Include source (e.g., Reuters, Bloomberg)
        - Include author name
        - Include article category (e.g., Earnings, M&A, Industry News)
        - Focus on {focus_area}
        - Maintain a {sentiment} tone

        Format the response as:
        HEADLINE: [your headline]
        SOURCE: [news source]
        AUTHOR: [author name]
        CATEGORY: [article category]
        ARTICLE: [your article]"""

        focus_areas = [
            'quarterly earnings',
            'market expansion',
            'product launch',
            'strategic partnership',
            'industry trends',
            'leadership changes',
            'merger and acquisition',
            'regulatory compliance',
            'market performance',
            'sustainability initiatives'
        ]
        
        news_sources = [
            'Reuters',
            'Bloomberg',
            'Financial Times',
            'Wall Street Journal',
            'CNBC',
            'MarketWatch'
        ]
        
        sentiments = ['positive', 'neutral', 'slightly negative', 'optimistic']

        # Using nested tqdm for detailed progress
        for comp_id in tqdm(self.company_ids, desc="Companies processed"):
            for _ in tqdm(range(self.n_articles_per_company), 
                         desc=f"Articles for {comp_id}", 
                         leave=False):
                try:
                    prompt = article_prompt.format(
                        company_id=comp_id,
                        focus_area=random.choice(focus_areas),
                        sentiment=random.choice(sentiments)
                    )
                    
                    response = self._generate_llm_response(prompt)
                    
                    # Parse the response
                    headline = ""
                    article = ""
                    source = ""
                    author = ""
                    category = ""
                    
                    for line in response.split('\n'):
                        if line.startswith('HEADLINE:'):
                            headline = line.replace('HEADLINE:', '').strip()
                        elif line.startswith('SOURCE:'):
                            source = line.replace('SOURCE:', '').strip()
                        elif line.startswith('AUTHOR:'):
                            author = line.replace('AUTHOR:', '').strip()
                        elif line.startswith('CATEGORY:'):
                            category = line.replace('CATEGORY:', '').strip()
                        elif line.startswith('ARTICLE:'):
                            article = line.replace('ARTICLE:', '').strip()
                    
                    if not headline or not article:
                        raise ValueError("Failed to parse article response")
                    
                    # Generate random date within last 2 years
                    date = datetime.now() - timedelta(
                        days=random.randint(1, 730)
                    )
                    
                    # Generate URL based on source and headline
                    url_headline = headline.lower().replace(' ', '-')
                    url = f"https://www.{source.lower().replace(' ', '')}.com/news/{date.strftime('%Y/%m/%d')}/{url_headline}"
                    
                    data.append({
                        'article_id': f'ART{str(article_id).zfill(6)}',
                        'company_id': comp_id,
                        'date': date.strftime('%Y-%m-%d'),
                        'headline': headline,
                        'text': article,
                        'source': source or random.choice(news_sources),
                        'author': author or "Financial Reporter",
                        'category': category or random.choice([
                            'Earnings',
                            'Market News',
                            'Industry Analysis',
                            'Corporate News',
                            'M&A',
                            'Market Analysis'
                        ]),
                        'url': url,
                        'word_count': len(article.split()),
                        'publish_time': f"{date.strftime('%Y-%m-%d')} {random.randint(8,20):02d}:{random.randint(0,59):02d}:00"
                    })
                    
                    logger.debug(f"Generated article {article_id} for {comp_id}")
                    article_id += 1
                    time.sleep(0.5)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to generate article for {comp_id}: {str(e)}")
                    # Add fallback data if generation fails
                    fallback_date = datetime.now()
                    data.append({
                        'article_id': f'ART{str(article_id).zfill(6)}',
                        'company_id': comp_id,
                        'date': fallback_date.strftime('%Y-%m-%d'),
                        'headline': f"Company {comp_id} Reports Quarterly Results",
                        'text': "The company reported its regular quarterly results today.",
                        'source': random.choice(news_sources),
                        'author': "Financial Reporter",
                        'category': "Earnings",
                        'url': f"https://www.reuters.com/news/{fallback_date.strftime('%Y/%m/%d')}/company-{comp_id}-reports",
                        'word_count': 10,
                        'publish_time': f"{fallback_date.strftime('%Y-%m-%d')} 09:00:00"
                    })
                    article_id += 1

        df = pd.DataFrame(data)
        logger.info(f"✅ Successfully generated {len(data)} news articles")
        return df
    
    def generate_financial_metrics(self) -> pd.DataFrame:
        """Generate financial metrics using Llama2 for trend analysis"""
        logger.info("Starting financial metrics generation...")
        data = []
        
        metrics_prompt = """Generate realistic financial metrics for a {size} company in the {industry} industry.
        Return ONLY a JSON with these metrics (numbers only, no text):
        - revenue (in millions)
        - debt_ratio (between 0 and 1)
        - current_ratio (typically between 1 and 3)
        - quick_ratio (typically between 0.5 and 2)
        - roa (return on assets, between 0 and 0.2)
        - roe (return on equity, between 0 and 0.3)
        - operating_margin (between 0 and 0.3)
        - ebitda_margin (between 0 and 0.4)
        - net_profit_margin (between 0 and 0.25)
        - asset_turnover (between 0.5 and 2.5)"""

        company_profiles = self.generate_company_profiles()
        
        for comp_id in tqdm(self.company_ids, desc="Generating financial metrics"):
            try:
                company = company_profiles[
                    company_profiles['company_id'] == comp_id
                ].iloc[0]
                
                # Get base metrics for the company
                prompt = metrics_prompt.format(
                    size=company['size'],
                    industry=company['industry']
                )
                
                response = self._generate_llm_response(prompt)
                base_metrics = eval(response.strip().replace("```json", "").replace("```", ""))
                
                # Generate quarterly data with realistic variations
                for quarter in range(8):  # 8 quarters = 2 years
                    date = datetime.now() - timedelta(days=90*quarter)
                    
                    # Add some random variation to metrics
                    metrics = {
                        key: value * random.uniform(0.95, 1.05)
                        for key, value in base_metrics.items()
                    }
                    
                    # Add calculated metrics
                    metrics['interest_coverage_ratio'] = metrics['operating_margin'] / max(metrics['debt_ratio'], 0.01)
                    metrics['debt_to_equity'] = metrics['debt_ratio'] / (1 - metrics['debt_ratio'])
                    
                    data.append({
                        'company_id': comp_id,
                        'date': date.strftime('%Y-%m-%d'),
                        **metrics
                    })
                
                logger.debug(f"Generated metrics for {comp_id}")
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate metrics for {comp_id}: {str(e)}")
                # Add fallback data if generation fails
                for quarter in range(8):
                    data.append({
                        'company_id': comp_id,
                        'date': (datetime.now() - timedelta(days=90*quarter)).strftime('%Y-%m-%d'),
                        'revenue': random.uniform(100, 1000),
                        'debt_ratio': random.uniform(0.2, 0.6),
                        'current_ratio': random.uniform(1.2, 2.5),
                        'quick_ratio': random.uniform(0.8, 1.8),
                        'roa': random.uniform(0.05, 0.15),
                        'roe': random.uniform(0.1, 0.25),
                        'operating_margin': random.uniform(0.08, 0.2),
                        'ebitda_margin': random.uniform(0.1, 0.3),
                        'net_profit_margin': random.uniform(0.05, 0.15),
                        'asset_turnover': random.uniform(0.8, 2.0),
                        'interest_coverage_ratio': random.uniform(2.0, 5.0),
                        'debt_to_equity': random.uniform(0.3, 1.5)
                    })

        df = pd.DataFrame(data)
        logger.info(f"✅ Successfully generated financial metrics for {self.n_companies} companies")
        return df

    def generate_credit_ratings(self) -> pd.DataFrame:
        """Generate credit ratings using financial metrics"""
        logger.info("Starting credit ratings generation...")
        data = []
        financial_data = self.generate_financial_metrics()
        
        rating_prompt = """Based on these financial metrics, suggest a credit rating and outlook:
        Revenue: {revenue}M
        Debt Ratio: {debt_ratio:.2f}
        Interest Coverage: {interest_coverage:.2f}
        ROE: {roe:.2f}
        Operating Margin: {operating_margin:.2f}
        Industry: {industry}
        Size: {size}

        Return ONLY a JSON with:
        - rating (AAA, AA+, AA, AA-, A+, A, A-, BBB+, BBB, BBB-, BB+, BB, BB-)
        - outlook (stable, positive, negative, watch)
        - rationale (brief explanation)
        """
        
        company_profiles = self.generate_company_profiles()
        
        for comp_id in tqdm(self.company_ids, desc="Generating credit ratings"):
            try:
                # Get company's latest financial metrics
                company_financials = financial_data[
                    financial_data['company_id'] == comp_id
                ].iloc[0]
                
                company_profile = company_profiles[
                    company_profiles['company_id'] == comp_id
                ].iloc[0]
                
                # Generate base rating
                prompt = rating_prompt.format(
                    revenue=round(company_financials['revenue'], 2),
                    debt_ratio=company_financials['debt_ratio'],
                    interest_coverage=company_financials.get('interest_coverage_ratio', 2.5),
                    roe=company_financials['roe'],
                    operating_margin=company_financials['operating_margin'],
                    industry=company_profile['industry'],
                    size=company_profile['size']
                )
                
                response = self._generate_llm_response(prompt)
                rating_data = eval(response.strip().replace("```json", "").replace("```", ""))
                
                # Generate 2-4 rating updates with dates
                n_updates = random.randint(2, 4)
                base_rating_idx = self.ratings.index(rating_data['rating'])
                
                # Generate dates ensuring they're in order
                dates = sorted([
                    datetime.now() - timedelta(days=random.randint(1, 730))
                    for _ in range(n_updates)
                ])
                
                prev_rating_idx = base_rating_idx
                for i, date in enumerate(dates):
                    # Allow ratings to change slightly over time
                    rating_idx = base_rating_idx + random.randint(-1, 1)
                    rating_idx = max(0, min(rating_idx, len(self.ratings) - 1))
                    
                    # Maybe change outlook occasionally
                    current_outlook = rating_data['outlook'] if random.random() > 0.3 else random.choice(self.outlooks)
                    
                    # Generate rationale based on changes
                    if i > 0 and rating_idx != prev_rating_idx:
                        change_type = "upgrade" if rating_idx < prev_rating_idx else "downgrade"
                        rationale = f"Rating {change_type} reflects changes in financial metrics and market conditions"
                    else:
                        rationale = rating_data.get('rationale', "Regular rating review")
                    
                    data.append({
                        'company_id': comp_id,
                        'date': date.strftime('%Y-%m-%d'),
                        'rating': self.ratings[rating_idx],
                        'outlook': current_outlook,
                        'rationale': rationale,
                        'previous_rating': self.ratings[prev_rating_idx] if i > 0 else None,
                        'review_type': 'Regular Review' if i == 0 else 'Rating Action'
                    })
                    
                    prev_rating_idx = rating_idx
                
                logger.debug(f"Generated ratings for {comp_id}")
                time.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate ratings for {comp_id}: {str(e)}")
                # Add fallback data if generation fails
                for _ in range(random.randint(2, 4)):
                    data.append({
                        'company_id': comp_id,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'rating': random.choice(self.ratings),
                        'outlook': random.choice(self.outlooks),
                        'rationale': "Regular rating review",
                        'previous_rating': None,
                        'review_type': 'Regular Review'
                    })

        df = pd.DataFrame(data)
        logger.info(f"✅ Successfully generated credit ratings")
        return df
    
    def generate_all_data(self, output_dir='data') -> Dict[str, pd.DataFrame]:
        """Generate all synthetic datasets"""
        start_time = time.time()
        logger.info("🚀 Starting synthetic data generation process...")
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Generate all datasets with timing information
        logger.info("\n📊 Generating company profiles...")
        t0 = time.time()
        company_profiles = self.generate_company_profiles()
        logger.info(f"Time taken: {time.time() - t0:.2f} seconds")
        
        logger.info("\n📰 Generating news articles...")
        t0 = time.time()
        news_articles = self.generate_news_articles()
        logger.info(f"Time taken: {time.time() - t0:.2f} seconds")
        
        logger.info("\n💹 Generating financial metrics...")
        t0 = time.time()
        financial_metrics = self.generate_financial_metrics()
        logger.info(f"Time taken: {time.time() - t0:.2f} seconds")
        
        logger.info("\n⭐ Generating credit ratings...")
        t0 = time.time()
        credit_ratings = self.generate_credit_ratings()
        logger.info(f"Time taken: {time.time() - t0:.2f} seconds")
        
        # Save files
        logger.info("\n💾 Saving generated data...")
        company_profiles.to_csv(f'{output_dir}/company_profiles.csv', index=False)
        news_articles.to_csv(f'{output_dir}/news_articles.csv', index=False)
        credit_ratings.to_csv(f'{output_dir}/credit_ratings.csv', index=False)
        financial_metrics.to_csv(f'{output_dir}/financial_metrics.csv', index=False)
        
        # Print summary statistics
        logger.info("\n📈 Generation Summary:")
        datasets = {
            'Company Profiles': company_profiles,
            'News Articles': news_articles,
            'Financial Metrics': financial_metrics,
            'Credit Ratings': credit_ratings
        }
        
        for name, df in datasets.items():
            logger.info(f"\n{name}:")
            logger.info(f"  Records: {len(df):,}")
            logger.info(f"  Columns: {len(df.columns)}")
            logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        total_time = time.time() - start_time
        logger.info(f"\n✨ Data generation completed in {total_time:.2f} seconds")
        
        return datasets