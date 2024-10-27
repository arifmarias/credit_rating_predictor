import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from src.data_generation.llama_generator import LlamaDataGenerator

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate synthetic financial data using Llama2'
    )
    parser.add_argument(
        '--companies',
        type=int,
        default=10,
        help='Number of companies to generate (default: 100)'
    )
    parser.add_argument(
        '--articles',
        type=int,
        default=5,
        help='Number of articles per company (default: 5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data',
        help='Output directory for generated files (default: data)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='llama2',
        help='Ollama model to use (default: llama2)'
    )
    return parser.parse_args()

def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / f'data_generation_{timestamp}.log')
        ]
    )
    return logging.getLogger(__name__)

def check_ollama():
    """Check if Ollama is running and model is available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code != 200:
            raise ConnectionError("Ollama is not responding correctly")
        return True
    except Exception as e:
        print(f"""
        Error: Could not connect to Ollama.
        Please ensure:
        1. Ollama is installed
        2. Ollama service is running (run 'ollama serve')
        3. Llama2 model is pulled (run 'ollama pull llama2')
        
        Error details: {str(e)}
        """)
        return False

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Setup directories
    project_root = Path(__file__).parents[1]
    log_dir = project_root / 'logs'
    output_dir = project_root / args.output
    
    # Setup logging
    logger = setup_logging(log_dir)
    logger.info("Starting data generation process...")
    
    # Check Ollama
    if not check_ollama():
        logger.error("Ollama check failed. Exiting.")
        return
    
    try:
        # Create output directory
        output_dir.mkdir(exist_ok=True)
        
        # Initialize generator
        generator = LlamaDataGenerator(
            n_companies=args.companies,
            n_articles_per_company=args.articles,
            model_name=args.model
        )
        
        # Generate data
        logger.info(f"Generating data for {args.companies} companies...")
        datasets = generator.generate_all_data(output_dir=str(output_dir))
        
        # Print summary
        logger.info("\nData Generation Summary:")
        for name, df in datasets.items():
            logger.info(f"\n{name}:")
            logger.info(f"Total records: {len(df)}")
            logger.info(f"Columns: {df.columns.tolist()}")
        
        logger.info(f"\nData files saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during data generation: {str(e)}")
        raise

if __name__ == "__main__":
    main()